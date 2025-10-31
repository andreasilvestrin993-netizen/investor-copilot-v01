import os, io, math, json, time, re, datetime as dt
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from docx import Document
from dotenv import load_dotenv

# ----------------------------
# Bootstrap
# ----------------------------
load_dotenv()
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = (APP_DIR / "data"); DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = (APP_DIR / "output"); OUTPUT_DIR.mkdir(exist_ok=True)

def read_yaml(path: Path):
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
CFG = read_yaml(APP_DIR / "config.yaml")

BASE_CCY = "EUR"  # hard-enforced base currency display

# Daily cache files for prices and FX rates
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
PRICES_CACHE_FILE = CACHE_DIR / "prices_cache.json"
FX_CACHE_FILE = CACHE_DIR / "fx_cache.json"

def load_daily_cache(cache_file: Path):
    """Load cache if it's from today, otherwise return empty dict"""
    if not cache_file.exists():
        return {}
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if cache is from today
        cache_date = data.get('date')
        today = datetime.now().date().isoformat()
        
        if cache_date == today:
            return data.get('data', {})
        else:
            return {}
    except Exception:
        return {}

def save_daily_cache(cache_file: Path, data: dict):
    """Save cache with today's date"""
    try:
        cache_data = {
            'date': datetime.now().date().isoformat(),
            'data': data
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        pass  # Silent fail for cache save errors

# Session state for performance caching
if 'symbol_cache' not in st.session_state:
    st.session_state.symbol_cache = {}
if 'prices_cache' not in st.session_state:
    st.session_state.prices_cache = load_daily_cache(PRICES_CACHE_FILE)
if 'last_price_fetch' not in st.session_state:
    # Set to today if we loaded cache, otherwise None
    if st.session_state.prices_cache:
        st.session_state.last_price_fetch = datetime.now()
    else:
        st.session_state.last_price_fetch = None
if 'fx_cache' not in st.session_state:
    st.session_state.fx_cache = load_daily_cache(FX_CACHE_FILE)
if 'last_fx_fetch' not in st.session_state:
    # Set to today if we loaded cache, otherwise None
    if st.session_state.fx_cache:
        st.session_state.last_fx_fetch = datetime.now()
    else:
        st.session_state.last_fx_fetch = None

# Session state for column mapping
if 'pending_upload' not in st.session_state:
    st.session_state.pending_upload = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'upload_type' not in st.session_state:
    st.session_state.upload_type = None
if 'csv_format_info' not in st.session_state:
    st.session_state.csv_format_info = None

# ----------------------------
# Formatting utilities
# ----------------------------
def format_price(value, currency="EUR"):
    """Format price to 2 decimals with currency symbol"""
    if value is None or pd.isna(value):
        return "-"
    return f"€{value:.2f}" if currency == "EUR" else f"{value:.2f}"

def format_percentage(value):
    """Format percentage to 1 decimal"""
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.1f}%"

def format_large_number(value):
    """Format large numbers with M/K suffixes"""
    if value is None or pd.isna(value):
        return "-"
    
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"€{value/1_000_000:.1f}M"
    elif abs_value >= 1_000:
        return f"€{value/1_000:.1f}K"
    else:
        return f"€{value:.2f}"

# ----------------------------
# Files (CSV "DB")
# ----------------------------
IND_GROWTH_CSV = DATA_DIR / "industry_growth.csv"
PORTFOLIO_CSV   = DATA_DIR / "portfolio.csv"
WATCHLIST_CSV   = DATA_DIR / "watchlists.csv"
OVERRIDES_CSV   = DATA_DIR / "symbol_overrides.csv"

PORTFOLIO_COLS = ["Name","Symbol","Quantity","BEP","Sector","Currency"]
WATCHLIST_COLS = ["Name","Symbol","Sector","Currency","Buy_Low","Buy_High","Target_1Y","Target_3Y","Target_5Y","Target_10Y"]
OVERRIDE_COLS  = ["UserSymbol","ProviderSymbol","ProviderCurrency"]

# ----------------------------
# Seed Industry->Sector growth table on first run
# ----------------------------
if not IND_GROWTH_CSV.exists():
    seed = pd.DataFrame([
        # Technology (≤10 industries total, ≤6 sectors each across whole app)
        ["Technology","AI & Machine Learning",30.0],
        ["Technology","Cloud Infrastructure & Services",27.0],
        ["Technology","Cybersecurity",15.0],
        ["Technology","Software (Applications & DevOps)",12.5],
        ["Technology","Semiconductors",10.5],
        ["Technology","Data Infrastructure (Storage, Networking, CDNs)",23.0],
        # Energy & Clean Tech
        ["Energy & Clean Tech","EV & Battery Systems",12.0],
        ["Energy & Clean Tech","Solar & Wind Power",11.0],
        ["Energy & Clean Tech","Hydrogen & Fuel Cells",8.0],
        ["Energy & Clean Tech","Nuclear Energy (incl. SMRs, Uranium)",9.0],
        ["Energy & Clean Tech","Utilities & Grid Tech",5.0],
        ["Energy & Clean Tech","Renewable Infrastructure",8.5],
        # Industrials
        ["Industrials","Aerospace & Defense",11.0],
        ["Industrials","Advanced Manufacturing & Equipment",7.5],
        ["Industrials","Construction & Engineering",4.5],
        ["Industrials","Security & Infrastructure",6.5],
        ["Industrials","Water & Environmental Systems",5.0],
        # Automotive & Mobility
        ["Automotive & Mobility","EV Manufacturers",11.0],
        ["Automotive & Mobility","Autonomous Vehicles & Lidar",14.0],
        ["Automotive & Mobility","Urban Air Mobility (eVTOL, Drones)",15.0],
        ["Automotive & Mobility","Telematics & Mobility Tech",10.0],
        # Financial Services
        ["Financial Services","Fintech & Neo-Banks",10.0],
        ["Financial Services","Trading Platforms & Exchanges",8.0],
        ["Financial Services","Crypto Infrastructure",20.0],
        ["Financial Services","AI-based Financial Services",15.0],
        # Healthcare & Life Sciences
        ["Healthcare & Life Sciences","Biotechnology & Genomics",12.0],
        ["Healthcare & Life Sciences","Medical Devices & Diagnostics",8.5],
        ["Healthcare & Life Sciences","Healthcare Software & Analytics",10.0],
        ["Healthcare & Life Sciences","Pharmaceuticals",6.0],
        # Consumer Tech & Digital Media
        ["Consumer Tech & Digital Media","E-Commerce Platforms",9.5],
        ["Consumer Tech & Digital Media","Social Media & Content",4.0],
        ["Consumer Tech & Digital Media","Gaming & Interactive Media",5.0],
        ["Consumer Tech & Digital Media","Consumer Electronics",4.5],
        # Telecom & Connectivity
        ["Telecom & Connectivity","Communication Infrastructure",5.5],
        ["Telecom & Connectivity","Networking & 5G Tech",8.5],
        ["Telecom & Connectivity","Telecom Services",4.5],
        # Materials & Mining
        ["Materials & Mining","Lithium & Battery Materials",9.0],
        ["Materials & Mining","Rare Earths & Advanced Materials",7.0],
        ["Materials & Mining","Uranium & Nuclear Fuel Supply",8.5],
        # Real Estate & Infrastructure
        ["Real Estate & Infrastructure","Data Centers",10.0],
        ["Real Estate & Infrastructure","Smart Infrastructure",7.5],
        ["Real Estate & Infrastructure","Real Estate Tech & Services",5.5],
    ], columns=["Industry","Sector","YoY_Growth_%"])
    seed.to_csv(IND_GROWTH_CSV, index=False)

# Get unique sectors from industry_growth
def get_sector_list():
    """Get list of all sectors from industry_growth.csv"""
    growth_df = load_csv(IND_GROWTH_CSV, ["Industry","Sector","YoY_Growth_%"])
    sectors = sorted(growth_df["Sector"].unique().tolist())
    return sectors

# Map industry/sector names to simplified categories
SECTOR_MAPPING = {
    "Technology": ["Technology", "AI & Machine Learning", "Cloud Infrastructure", "Software", "Semiconductors"],
    "Energy & Clean Tech": ["Energy", "Renewable", "Solar", "Wind", "Nuclear"],
    "Automotive & Mobility": ["Automotive", "EV", "Electric Vehicle", "Mobility"],
    "Financial Services": ["Finance", "Fintech", "Banking", "Insurance"],
    "Healthcare & Life Sciences": ["Healthcare", "Biotech", "Pharma", "Medical"],
    "Consumer Tech & Digital Media": ["Consumer", "E-Commerce", "Media", "Gaming"],
    "Telecom & Connectivity": ["Telecom", "Communication", "5G"],
    "Industrials": ["Industrial", "Manufacturing", "Aerospace"],
    "Materials & Mining": ["Materials", "Mining", "Metals"],
    "Real Estate & Infrastructure": ["Real Estate", "Infrastructure", "Construction"]
}

def map_industry_to_sector(industry_text):
    """Map Marketstack industry to our sector list"""
    if not industry_text:
        return None
    
    industry_lower = industry_text.lower()
    
    # Direct match first
    sectors = get_sector_list()
    for sector in sectors:
        if sector.lower() in industry_lower or industry_lower in sector.lower():
            return sector
    
    # Keyword mapping
    for sector, keywords in SECTOR_MAPPING.items():
        for keyword in keywords:
            if keyword.lower() in industry_lower:
                return sector
    
    return None

# Ensure overrides file exists
if not OVERRIDES_CSV.exists():
    pd.DataFrame(columns=OVERRIDE_COLS).to_csv(OVERRIDES_CSV, index=False)

# ----------------------------
# Helpers
# ----------------------------
def get_secret(name, default=""):
    return os.getenv(name, default)

def clamp_format_number(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    n = float(x); absn = abs(n)
    if absn >= 1_000_000_000: return f"{n/1_000_000_000:.0f}B"
    if absn >= 1_000_000:     return f"{n/1_000_000:.0f}M"
    if absn >= 1_000:         return f"{n/1_000:.0f}K"
    return f"{int(round(n,0))}"

def fmt_pct(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.1f}%"

# ----------------------------
# CSV Format Detection & Parsing
# ----------------------------
def detect_csv_format(file_content):
    """
    Detect if CSV uses European format (semicolon delimiter, comma decimal separator)
    or standard format (comma delimiter, dot decimal separator).
    Returns (delimiter, decimal_separator)
    """
    # Read first few lines to detect format
    first_lines = file_content[:2000]  # Check first 2000 chars
    
    # Count delimiters
    semicolon_count = first_lines.count(';')
    comma_count = first_lines.count(',')
    
    # European format typically has more semicolons than commas in header/data
    # (semicolons separate columns, commas are in decimal numbers)
    if semicolon_count > comma_count:
        return ';', ','  # European format
    else:
        return ',', '.'  # Standard format

def read_csv_smart(uploaded_file):
    """
    Smart CSV reader that handles both European and standard formats.
    Returns DataFrame with properly parsed numeric values.
    """
    # Read file content
    file_content = uploaded_file.getvalue().decode('utf-8')
    
    # Detect format
    delimiter, decimal_sep = detect_csv_format(file_content)
    
    # Reset file pointer
    uploaded_file.seek(0)
    
    # Read CSV with detected format - explicitly set index_col=False to prevent first column being used as index
    if decimal_sep == ',':
        # European format: need to convert commas to dots in numeric columns
        df = pd.read_csv(uploaded_file, sep=delimiter, dtype=str, na_values=['', 'None', 'NaN', 'null'], index_col=False)
        
        # Convert numeric columns: replace comma with dot
        for col in df.columns:
            # Try to detect if column contains numeric data with commas
            if df[col].dtype == 'object':
                # Check if values look like European decimals (e.g., "123,45")
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    # If any value contains comma followed by digits, treat as European decimal
                    if any(pd.notna(val) and isinstance(val, str) and re.search(r',\d', val) for val in sample):
                        df[col] = df[col].str.replace(',', '.', regex=False)
    else:
        # Standard format
        df = pd.read_csv(uploaded_file, sep=delimiter, dtype=str, na_values=['', 'None', 'NaN', 'null'], index_col=False)
    
    return df, delimiter, decimal_sep

# ----------------------------
# Column Mapping UI
# ----------------------------
def show_column_mapping_ui(uploaded_df, expected_cols, csv_type="portfolio", csv_format_info=None):
    """
    Show column mapping interface for CSV uploads.
    Returns mapped DataFrame if confirmed, None if still mapping.
    """
    st.info(f"📋 **Column Mapping for {csv_type.title()}**")
    
    # Show detected format
    if csv_format_info:
        delimiter, decimal_sep = csv_format_info
        format_type = "European (semicolon delimiter, comma decimal)" if delimiter == ';' else "Standard (comma delimiter, dot decimal)"
        st.success(f"✓ Detected CSV format: **{format_type}**")
    
    st.write("Map your CSV columns to the expected format:")
    
    # Get uploaded CSV columns
    uploaded_cols = uploaded_df.columns.tolist()
    
    # Create mapping interface
    mapping = {}
    st.write("**Your CSV columns → Expected columns:**")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.write("**Your CSV Column**")
    with col2:
        st.write("**→**")
    with col3:
        st.write("**Maps to Expected Column**")
    
    # Create mapping dropdowns for each expected column
    for i, expected_col in enumerate(expected_cols):
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            # Try to find best match
            best_match = None
            expected_lower = expected_col.lower().replace("_", "").replace(" ", "")
            
            for uploaded_col in uploaded_cols:
                uploaded_lower = uploaded_col.lower().replace("_", "").replace(" ", "")
                if expected_lower == uploaded_lower:
                    best_match = uploaded_col
                    break
            
            # If no exact match, try partial match
            if not best_match:
                for uploaded_col in uploaded_cols:
                    uploaded_lower = uploaded_col.lower().replace("_", "").replace(" ", "")
                    if expected_lower in uploaded_lower or uploaded_lower in expected_lower:
                        best_match = uploaded_col
                        break
            
            # Default to first column if still no match
            if not best_match and uploaded_cols:
                best_match = uploaded_cols[0] if i < len(uploaded_cols) else uploaded_cols[0]
            
            default_idx = uploaded_cols.index(best_match) if best_match in uploaded_cols else 0
            
            selected = st.selectbox(
                f"Column for '{expected_col}'",
                options=["(skip)"] + uploaded_cols,
                index=default_idx + 1 if best_match else 0,
                key=f"map_{csv_type}_{i}",
                label_visibility="collapsed"
            )
            
            if selected != "(skip)":
                mapping[expected_col] = selected
        
        with col2:
            st.write("→")
        
        with col3:
            st.write(f"**{expected_col}**")
    
    # Create preview DataFrame based on current mapping
    preview_df = pd.DataFrame()
    for expected_col, uploaded_col in mapping.items():
        if uploaded_col in uploaded_df.columns:
            preview_df[expected_col] = uploaded_df[uploaded_col]
    
    # Fill missing columns with empty values for preview
    for col in expected_cols:
        if col not in preview_df.columns:
            preview_df[col] = None
    
    # Show live preview with mapped columns
    with st.expander("📊 Preview mapped data (first 5 rows)", expanded=True):
        st.caption("This preview updates as you change the column mappings above")
        st.dataframe(preview_df.head(), use_container_width=True)
    
    # Action buttons
    col_confirm, col_cancel = st.columns(2)
    
    with col_confirm:
        if st.button("✅ Confirm Mapping", type="primary", use_container_width=True):
            return preview_df
    
    with col_cancel:
        if st.button("❌ Cancel Upload", use_container_width=True):
            st.session_state.pending_upload = None
            st.session_state.upload_type = None
            st.session_state.csv_format_info = None
            st.rerun()
    
    return None

def fmt_price(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.2f}"

def ensure_columns(df: pd.DataFrame, cols: list):
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

def load_csv(path, cols):
    if path.exists():
        try:
            # Try to read with semicolon delimiter first (European format)
            df = pd.read_csv(path, sep=';', index_col=False)
            # If only one column, try comma delimiter
            if len(df.columns) == 1:
                df = pd.read_csv(path, sep=',', index_col=False)
        except Exception:
            df = pd.DataFrame(columns=cols)
        return ensure_columns(df, cols)
    return pd.DataFrame(columns=cols)

def save_csv(path, df):
    # Save with semicolon delimiter (European format compatible)
    df.to_csv(path, index=False, sep=';')

def write_docx(text: str, out_path: Path):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(out_path.as_posix())

def guess_week_folder(dt_obj: date):
    monday = dt_obj - dt.timedelta(days=dt_obj.weekday())
    friday = monday + dt.timedelta(days=4)
    return f"{monday.day:02d}-{friday.day:02d} {friday.strftime('%B')} {friday.year}"

def month_folder(dt_obj: date):
    return f"{dt_obj.strftime('%m.%Y')}"

def load_growth_table():
    return pd.read_csv(IND_GROWTH_CSV)[["Industry","Sector","YoY_Growth_%"]]

def save_growth_table(df):
    df[["Industry","Sector","YoY_Growth_%"]].to_csv(IND_GROWTH_CSV, index=False)

# ----------------------------
# Marketstack FX & Prices (EOD) + Fallbacks
# ----------------------------
def ms_get(url, params):
    # Generic helper for Marketstack GET with query params
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def fetch_fx_map_eur(marketstack_key: str) -> dict:
    """
    Build FX map to EUR using Marketstack 'tickers/eod/latest' on currency crosses.
    We compute:
      USD->EUR  from EURUSD (so 1 USD = 1 / EURUSD close)
      GBP->EUR  from EURGBP (so 1 GBP = 1 / EURGBP close)
      CHF->EUR  from EURCHF (so 1 CHF = 1 / EURCHF close)
    Cached daily (only fetches once per day). Uses Frankfurter API as fallback.
    """
    # Check if cache is from today
    now = datetime.now()
    cache_is_today = (st.session_state.last_fx_fetch and 
                     st.session_state.last_fx_fetch.date() == now.date())
    
    if cache_is_today and st.session_state.fx_cache:
        return st.session_state.fx_cache
    
    fx = {"EUR": 1.0}
    
    def latest_fx(pair):
        """Fetch latest FX rate from Marketstack"""
        if not marketstack_key:
            return None
        try:
            base_url = "https://api.marketstack.com/v1/tickers/{}/eod/latest".format(pair)
            params = {"access_key": marketstack_key}
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                close = data.get("data", {}).get("close")
                return float(close) if close is not None else None
        except Exception:
            pass
        return None

    # Try Marketstack first
    eurusd = latest_fx("EURUSD")
    if eurusd and eurusd > 0:
        fx["USD"] = 1.0 / eurusd

    eurgbp = latest_fx("EURGBP")
    if eurgbp and eurgbp > 0:
        fx["GBP"] = 1.0 / eurgbp

    eurchf = latest_fx("EURCHF")
    if eurchf and eurchf > 0:
        fx["CHF"] = 1.0 / eurchf

    # Fallback to Frankfurter API for any missing rates
    missing = [k for k in ["USD", "GBP", "CHF"] if k not in fx]
    if missing:
        try:
            r = requests.get("https://api.frankfurter.app/latest?from=EUR", timeout=10)
            if r.status_code == 200:
                rates = r.json().get("rates", {})
                for ccy in missing:
                    if ccy in rates and rates[ccy] > 0:
                        # rates[ccy] = ccy per 1 EUR; want 1 ccy -> EUR => 1 / rates[ccy]
                        fx[ccy] = 1.0 / float(rates[ccy])
        except Exception:
            pass
    
    # Final fallback - use approximate rates only if still missing
    if "USD" not in fx:
        fx["USD"] = 0.92  # ~1.09 USD per EUR
    if "GBP" not in fx:
        fx["GBP"] = 0.85  # ~1.18 GBP per EUR
    if "CHF" not in fx:
        fx["CHF"] = 1.05  # ~0.95 CHF per EUR

    # Derived rates
    fx["HKD"] = fx.get("USD", 0.92) / 7.8  # HKD pegged to USD
    fx["CAD"] = fx.get("USD", 0.92) / 1.35
    fx["SEK"] = fx.get("EUR", 1.0) / 11.5
    fx["DKK"] = fx.get("EUR", 1.0) / 7.46
    fx["PLN"] = fx.get("EUR", 1.0) / 4.35
    
    # Cache the result and save to disk
    st.session_state.fx_cache = fx
    st.session_state.last_fx_fetch = now
    save_daily_cache(FX_CACHE_FILE, fx)
    
    return fx

def fetch_eod_prices(symbols, marketstack_key: str) -> dict:
    """Return dict {ProviderSymbol: close} using Marketstack EOD latest with daily caching."""
    if not symbols or not marketstack_key: 
        return {}
    
    # Check if cache is from today
    now = datetime.now()
    cache_is_today = (st.session_state.last_price_fetch and 
                     st.session_state.last_price_fetch.date() == now.date())
    
    if cache_is_today:
        # Return cached prices for requested symbols (from today)
        cached = {sym: st.session_state.prices_cache.get(sym) for sym in symbols 
                 if sym in st.session_state.prices_cache}
        if len(cached) == len([s for s in symbols if s]):  # All symbols found in cache
            return cached
    
    out = {}
    base_url = "http://api.marketstack.com/v1/eod/latest"
    # Marketstack allows batching via comma-separated symbols
    for i in range(0, len(symbols), 80):
        batch = symbols[i:i+80]
        data = ms_get(base_url, {"access_key": marketstack_key, "symbols": ",".join(batch)})
        
        # Handle API response - data should be a dict with "data" key containing list of items
        if not isinstance(data, dict):
            continue
            
        data_items = data.get("data", [])
        if not isinstance(data_items, list):
            continue
            
        for item in data_items:
            # Skip if item is not a dictionary
            if not isinstance(item, dict):
                continue
                
            sym = item.get("symbol")
            close = item.get("close")
            if sym and close is not None:
                price = float(close)
                out[sym] = price
                # Update cache
                st.session_state.prices_cache[sym] = price
        time.sleep(0.2)
    
    # Update cache timestamp and save to disk
    st.session_state.last_price_fetch = now
    save_daily_cache(PRICES_CACHE_FILE, st.session_state.prices_cache)
    return out

def fetch_52week_data(symbols, marketstack_key: str) -> dict:
    """
    Fetch 52-week high/low for symbols using Marketstack EOD endpoint.
    Returns dict {symbol: {"high": float, "low": float}}
    NOTE: This is cached for 24 hours to avoid slow API calls
    """
    if not symbols or not marketstack_key:
        return {}
    
    # Check cache (daily expiry for 52w data)
    cache_key = "week52_cache"
    cache_time_key = "last_52w_fetch"
    
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    
    now = datetime.now()
    if (cache_time_key in st.session_state and st.session_state[cache_time_key] and
        (now - st.session_state[cache_time_key]).total_seconds() < 86400):  # 24 hour cache
        cached = {sym: st.session_state[cache_key].get(sym) for sym in symbols 
                 if sym in st.session_state[cache_key]}
        if len(cached) == len([s for s in symbols if s]):
            return cached
    
    # For performance: only fetch 52-week data on-demand, not automatically
    # Return empty dict - will use current price as fallback
    return {}

def search_alternative_symbols(query: str, marketstack_key: str, limit=5):
    """
    Use Marketstack /tickers?search= to find possible provider symbols (alt listings).
    """
    base_url = "http://api.marketstack.com/v1/tickers"
    resp = ms_get(base_url, {"access_key": marketstack_key, "search": query, "limit": limit})
    return resp.get("data", [])

def pick_best_alt(symbol_row, name_hint=None):
    """
    Very simple ranking heuristic:
      - has_eod true
      - exchange MIC / name present (nice to have)
      - if name_hint provided, prefer item whose name contains hint tokens
    """
    if not symbol_row: return None
    if name_hint:
        name_hint_low = name_hint.lower()
        symbol_row = sorted(symbol_row, key=lambda x: 0 if ((x.get("name") or "").lower().find(name_hint_low) >= 0) else 1)
    # prefer has_eod
    symbol_row = sorted(symbol_row, key=lambda x: 0 if x.get("has_eod") else 1)
    return symbol_row[0]

def load_overrides():
    return load_csv(OVERRIDES_CSV, OVERRIDE_COLS)

def save_override(user_symbol: str, provider_symbol: str, provider_ccy: str):
    """Auto-save a discovered symbol override"""
    try:
        overrides = load_overrides()
        # Check if override already exists
        existing = overrides[overrides["UserSymbol"].fillna("").str.upper() == user_symbol.upper()]
        if existing.empty:
            # Add new override
            new_row = pd.DataFrame([{
                "UserSymbol": user_symbol,
                "ProviderSymbol": provider_symbol,
                "ProviderCurrency": provider_ccy
            }])
            overrides = pd.concat([overrides, new_row], ignore_index=True)
            save_csv(OVERRIDES_CSV, overrides)
            return True
    except Exception as e:
        st.warning(f"Could not auto-save override for {user_symbol}: {str(e)}")
    return False

def resolve_provider_symbol(user_symbol: str, name_hint: str, ccy: str, marketstack_key: str, prices_cache: dict, auto_save: bool = False) -> tuple[str,str]:
    """
    Return (provider_symbol, provider_ccy)
    Priority:
      1) Manual override in symbol_overrides.csv (HIGHEST PRIORITY - always check first)
      2) Direct user_symbol if it prices
      3) Search by symbol; else search by name hint; pick best; use its 'symbol' (and auto-save if enabled)
    """
    # PRIORITY 1: Check overrides FIRST (before cache)
    overrides = load_overrides()
    ov = overrides[overrides["UserSymbol"].fillna("").str.upper() == (user_symbol or "").upper()]
    if not ov.empty:
        row = ov.iloc[0]
        psym = (row.get("ProviderSymbol") or "").strip()
        pccy = (row.get("ProviderCurrency") or ccy or "EUR").strip() or "EUR"
        return psym or user_symbol, pccy

    # PRIORITY 2: If the direct symbol already priced in cache, use it (only if no override)
    if user_symbol in prices_cache:
        return user_symbol, ccy or "EUR"

    # PRIORITY 3: Search by symbol first
    cand = search_alternative_symbols(user_symbol, marketstack_key, limit=5)
    best = pick_best_alt(cand, name_hint)
    if best and best.get("symbol"):
        found_symbol = best["symbol"]
        found_ccy = best.get("currency") or ccy or "EUR"
        # Auto-save the discovered mapping
        if auto_save and found_symbol != user_symbol:
            if save_override(user_symbol, found_symbol, found_ccy):
                st.success(f"✓ Auto-saved override: {user_symbol} → {found_symbol}")
        return found_symbol, found_ccy

    # Try by name hint if present
    if name_hint:
        cand2 = search_alternative_symbols(name_hint, marketstack_key, limit=5)
        best2 = pick_best_alt(cand2, name_hint)
        if best2 and best2.get("symbol"):
            found_symbol = best2["symbol"]
            found_ccy = best2.get("currency") or ccy or "EUR"
            # Auto-save the discovered mapping
            if auto_save and found_symbol != user_symbol:
                if save_override(user_symbol, found_symbol, found_ccy):
                    st.success(f"✓ Auto-saved override: {user_symbol} → {found_symbol} (by name)")
            return found_symbol, found_ccy

    # Give up: return original symbol
    return user_symbol, ccy or "EUR"

def search_companies(query: str, marketstack_key: str, limit=10):
    """
    Search for companies by name or symbol using Marketstack API.
    Returns list of dicts with: name, symbol, sector, currency
    """
    if not query or len(query) < 2:
        return []
    
    try:
        results = search_alternative_symbols(query, marketstack_key, limit=limit)
        
        companies = []
        for item in results:
            symbol = item.get("symbol", "")
            name = item.get("name", "")
            currency = item.get("currency", "EUR")
            
            # Try to map Marketstack stock_exchange info to our sector
            exchange_info = item.get("stock_exchange", {})
            industry = exchange_info.get("name", "") if isinstance(exchange_info, dict) else ""
            
            # Map industry to our sector list
            sector = map_industry_to_sector(industry)
            
            companies.append({
                "name": name,
                "symbol": symbol,
                "sector": sector,
                "currency": currency,
                "display": f"{name} ({symbol})"
            })
        
        return companies
    except Exception as e:
        st.error(f"Error searching companies: {e}")
        return []

def to_eur(amount, ccy, fx_map):
    """
    Central FX converter - converts any amount to EUR.
    
    Uses live Marketstack rates (cached 1h) with Frankfurter fallback.
    All portfolio calculations use this single function to ensure consistency.
    
    Args:
        amount: The amount to convert
        ccy: Currency code (USD, GBP, CHF, EUR, etc.)
        fx_map: FX rate map from fetch_fx_map_eur()
    
    Returns:
        Amount in EUR, or None if conversion not possible
    """
    if amount is None or pd.isna(amount): 
        return None
    if (not ccy) or ccy.upper() == "EUR": 
        return float(amount)
    
    rate = fx_map.get(ccy.upper(), None)
    return float(amount) * float(rate) if rate else None

# ----------------------------
# YouTube helpers (RSS + transcript; no Google API)
# ----------------------------
def is_video_url(url: str) -> bool:
    return ("youtube.com/watch" in url) or ("youtu.be/" in url)

def extract_video_id(url: str) -> str | None:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return None

def channel_rss_url(channel_id: str) -> str:
    return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

def resolve_channel_id(channel_url_or_handle: str) -> str | None:
    try:
        if channel_url_or_handle.startswith("UC"):
            return channel_url_or_handle
        url = channel_url_or_handle.strip()
        if not url.startswith("http"):
            url = f"https://www.youtube.com/{url.lstrip('/')}"
        r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        m = re.search(r'"channelId":"(UC[0-9A-Za-z_\-]{20,})"', r.text)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None

def fetch_latest_from_channel(channel_id: str, since_date: date) -> list[dict]:
    import xml.etree.ElementTree as ET
    url = channel_rss_url(channel_id)
    vids = []
    all_vids = []
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text
            link = entry.find('atom:link', ns).attrib.get('href')
            published = entry.find('atom:published', ns).text
            pub_date = date.fromisoformat(published[:10])
            video_data = {'title': title, 'url': link, 'published': published}
            all_vids.append(video_data)
            
            # Get videos from the last 7 days
            days_diff = (since_date - pub_date).days
            if 0 <= days_diff <= 7:
                vids.append(video_data)
        
        # If no recent videos found, get the latest 3 videos as baseline
        if not vids and all_vids:
            vids = all_vids[:3]  # RSS feeds are typically ordered by date descending
            
    except Exception:
        pass
    return vids

def fetch_transcript_text(video_id: str) -> str | None:
    try:
        parts = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([p["text"] for p in parts if p.get("text")])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None

# ----------------------------
# OpenAI summarization
# ----------------------------
def summarize_texts(texts: list[str], openai_key: str, date_str: str, mode="daily"):
    if not openai_key:
        return f"[{mode.upper()} {date_str}] OpenAI key missing. Paste it in the sidebar."
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)
    prompt = f"""
You are an investment analyst. Summarize the following sources into a single {mode} brief for {date_str}.
Output sections in this exact order with short bullet points:

1) One-paragraph market summary
2) Macro trend & expectations (up / down / sideways, key catalysts)
3) Top news (most relevant 5)
4) Spotlights (stocks, sectors, events) with 1–2 bullets each
5) Stock picks: 
   - Watch: 3–5 tickers + one-line reason 
   - Buy: 1–3 tickers + entry rationale and risk

Be concise and practical.
"""
    joined = "\n\n".join([f"Source {i+1}:\n{t[:8000]}" for i,t in enumerate(texts if texts else ["(no sources)"])])
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role":"system","content":"Be concise, actionable, neutral."},
            {"role":"user","content": prompt + "\n\n" + joined}
        ]
    )
    return chat.choices[0].message.content.strip()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Investor Copilot — EUR EOD", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Investor Copilot — EUR (EOD, auto-FX, alt listings)")

# Sidebar (keys)
st.sidebar.title("API Keys")
MARKETSTACK_KEY = st.sidebar.text_input("Marketstack API key", value=get_secret("MARKETSTACK_KEY"), type="password")
OPENAI_KEY = st.sidebar.text_input("OpenAI API key", value=get_secret("OPENAI_API_KEY"), type="password")

# Auto FX fetch with error handling
try:
    FX_MAP = fetch_fx_map_eur(MARKETSTACK_KEY)
    usd_rate = FX_MAP.get("USD")
    gbp_rate = FX_MAP.get("GBP")
    chf_rate = FX_MAP.get("CHF")
    
    cached = st.session_state.last_fx_fetch and (time.time() - st.session_state.last_fx_fetch < 3600)
    cache_status = "✓ Cached" if cached else "🔄 Live"
    
    if usd_rate and gbp_rate and chf_rate:
        st.sidebar.success(f"💱 FX Rates {cache_status}")
        st.sidebar.caption(f"1 USD = €{usd_rate:.4f}")
        st.sidebar.caption(f"1 GBP = €{gbp_rate:.4f}")
        st.sidebar.caption(f"1 CHF = €{chf_rate:.4f}")
    else:
        st.sidebar.warning(f"⚠️ FX rates loaded with fallback")
        if usd_rate:
            st.sidebar.caption(f"1 USD = €{usd_rate:.4f}")
        if gbp_rate:
            st.sidebar.caption(f"1 GBP = €{gbp_rate:.4f}")
        if chf_rate:
            st.sidebar.caption(f"1 CHF = €{chf_rate:.4f}")
except Exception as e:
    # Emergency fallback - fetch_fx_map_eur should always return something
    FX_MAP = fetch_fx_map_eur("")
    st.sidebar.error(f"⚠️ FX error: {str(e)}")
    st.sidebar.caption(f"Using emergency fallback rates")

tabs = st.tabs(["🏠 Dashboard","💼 Portfolio","🔭 Watchlists","📰 Analysis","⚙️ Settings"])

# ----------------------------
# Dashboard
# ----------------------------
with tabs[0]:
    st.subheader("Overview (EUR)")
    port = load_csv(PORTFOLIO_CSV, PORTFOLIO_COLS)
    overrides = load_overrides()

    if port.empty:
        st.info("No portfolio yet. Add in the Portfolio tab or import a CSV.")
    else:
        # FX ARCHITECTURE:
        # 1. Fetch all stock prices in their native currencies
        # 2. Convert ONCE to EUR using to_eur(price, currency, FX_MAP)
        # 3. Calculate all totals/charts from EUR values only (never reconvert)
        # 4. FX_MAP contains live rates from Marketstack (1h cache) with Frankfurter fallback
        
        # Build provider symbols with overrides FIRST - optimized batch approach
        overrides = load_overrides()
        symbols_to_fetch = []
        symbol_map = {}  # user_symbol -> provider_symbol
        
        for user_sym in port["Symbol"].dropna().unique():
            user_sym_str = str(user_sym)
            # Check if there's an override
            ov = overrides[overrides["UserSymbol"].fillna("").str.upper() == user_sym_str.upper()]
            if not ov.empty:
                provider_sym = (ov.iloc[0].get("ProviderSymbol") or "").strip()
                if provider_sym:
                    symbols_to_fetch.append(provider_sym)
                    symbol_map[user_sym_str] = provider_sym
                else:
                    symbols_to_fetch.append(user_sym_str)
                    symbol_map[user_sym_str] = user_sym_str
            else:
                symbols_to_fetch.append(user_sym_str)
                symbol_map[user_sym_str] = user_sym_str
        
        # Fetch all prices using provider symbols
        prices_direct = fetch_eod_prices(symbols_to_fetch, MARKETSTACK_KEY) if MARKETSTACK_KEY else {}

        total_val_eur = 0.0
        total_cost_eur = 0.0
        
        # Debug: Track conversions
        debug_info = []

        for _, r in port.iterrows():
            qty = float(r["Quantity"] or 0)
            bep = float(r["BEP"] or 0.0)
            user_ccy = str(r["Currency"] or "EUR").upper() if pd.notna(r["Currency"]) else "EUR"
            user_sym = str(r["Symbol"]) if pd.notna(r["Symbol"]) else ""
            name_hint = str(r["Name"] or "") if pd.notna(r["Name"]) else ""

            # Get provider symbol from map (already resolved with overrides)
            psym = symbol_map.get(user_sym, user_sym)
            
            # Check if we have override currency
            ov = overrides[overrides["UserSymbol"].fillna("").str.upper() == user_sym.upper()]
            if not ov.empty:
                pccy = (ov.iloc[0].get("ProviderCurrency") or user_ccy).strip() or "EUR"
            else:
                pccy = user_ccy

            # Read price and convert
            last = prices_direct.get(psym, None)
            last_eur = to_eur(last, pccy, FX_MAP) if last is not None else None

            # Convert BEP (user native) to EUR
            bep_eur = to_eur(bep, user_ccy, FX_MAP) if bep else None

            if last_eur is not None and bep_eur is not None:
                value_eur = qty * last_eur
                total_val_eur += value_eur
                total_cost_eur += qty * bep_eur
                
                # Debug: Store conversion info
                debug_info.append({
                    "symbol": user_sym,
                    "qty": qty,
                    "price_native": last,
                    "price_ccy": pccy,
                    "price_eur": last_eur,
                    "fx_rate": FX_MAP.get(pccy, 1.0),
                    "value_eur": value_eur
                })

        total_pnl_eur = total_val_eur - total_cost_eur
        pnl_pct = (total_pnl_eur / total_cost_eur * 100) if total_cost_eur > 0 else 0.0

        # Save daily snapshot
        try:
            import sys
            sys.path.insert(0, str(APP_DIR / "utils"))
            from portfolio_history import save_daily_snapshot, load_portfolio_history
            save_daily_snapshot(total_val_eur, total_pnl_eur, pnl_pct, DATA_DIR)
        except:
            pass  # Silently fail if history module not available

        # Top metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value", format_large_number(total_val_eur))
        c2.metric("Total P/L", format_large_number(total_pnl_eur), format_percentage(pnl_pct))
        c3.metric("Positions", f"{len(port)}")
        c4.metric("Display Currency", BASE_CCY)

        st.divider()

        # Portfolio Breakdowns
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📊 Portfolio Composition")
            
            # Breakdown by Stock (Top 10)
            stock_breakdown = []
            for _, r in port.iterrows():
                qty = float(r["Quantity"] or 0)
                user_sym = str(r["Symbol"]) if pd.notna(r["Symbol"]) else ""
                name = str(r["Name"] or user_sym)
                user_ccy = str(r["Currency"] or "EUR").upper()
                
                # Use symbol_map to get provider symbol (already resolved with overrides)
                psym = symbol_map.get(user_sym, user_sym)
                
                # Get provider currency from override
                ov = overrides[overrides["UserSymbol"].fillna("").str.upper() == user_sym.upper()]
                if not ov.empty:
                    pccy = (ov.iloc[0].get("ProviderCurrency") or user_ccy).strip() or "EUR"
                else:
                    pccy = user_ccy
                
                last = prices_direct.get(psym)
                last_eur = to_eur(last, pccy, FX_MAP) if last is not None else None
                
                if last_eur:
                    value = qty * last_eur
                    pct = (value / total_val_eur * 100) if total_val_eur > 0 else 0
                    stock_breakdown.append({"Stock": name[:30], "Value": value, "% of Portfolio": pct})
            
            if stock_breakdown:
                stock_df_full = pd.DataFrame(stock_breakdown).sort_values("Value", ascending=False)
                
                # Top 10 + Other
                if len(stock_df_full) > 10:
                    top_10 = stock_df_full.head(10)
                    other_value = stock_df_full.iloc[10:]["Value"].sum()
                    other_pct = stock_df_full.iloc[10:]["% of Portfolio"].sum()
                    
                    other_row = pd.DataFrame([{
                        "Stock": "Other",
                        "Value": other_value,
                        "% of Portfolio": other_pct
                    }])
                    stock_df = pd.concat([top_10, other_row], ignore_index=True)
                else:
                    stock_df = stock_df_full
                
                st.caption("Top 10 Holdings by Portfolio %")
                
                # Pie chart with hover labels
                fig = px.pie(stock_df, values="% of Portfolio", names="Stock", 
                            title="",
                            hover_data={"Value": ":,.2f"},
                            color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent',  # Only show percentage inside
                    hovertemplate='<b>%{label}</b><br>€%{customdata[0]:,.2f}<br>%{percent}<extra></extra>'
                )
                fig.update_layout(showlegend=True, height=450, legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02
                ))
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("🏭 By Sector/Industry")
            
            # Sector breakdown
            sector_breakdown = {}
            for _, r in port.iterrows():
                qty = float(r["Quantity"] or 0)
                sector = str(r.get("Sector", "Unknown") or "Unknown")
                user_sym = str(r["Symbol"]) if pd.notna(r["Symbol"]) else ""
                user_ccy = str(r["Currency"] or "EUR").upper()
                
                # Use symbol_map to get provider symbol (already resolved with overrides)
                psym = symbol_map.get(user_sym, user_sym)
                
                # Get provider currency from override
                ov = overrides[overrides["UserSymbol"].fillna("").str.upper() == user_sym.upper()]
                if not ov.empty:
                    pccy = (ov.iloc[0].get("ProviderCurrency") or user_ccy).strip() or "EUR"
                else:
                    pccy = user_ccy
                
                last = prices_direct.get(psym)
                last_eur = to_eur(last, pccy, FX_MAP) if last is not None else None
                
                if last_eur:
                    value = qty * last_eur
                    if sector not in sector_breakdown:
                        sector_breakdown[sector] = 0
                    sector_breakdown[sector] += value
            
            if sector_breakdown:
                # Extract industry name (before parentheses/dash) and keep full detail
                sector_data = []
                for k, v in sector_breakdown.items():
                    # Extract base industry name
                    industry = k.split('(')[0].split('—')[0].strip()
                    sector_data.append({
                        "Industry": industry,
                        "Full_Sector": k,
                        "Value": v,
                        "% of Portfolio": (v / total_val_eur * 100) if total_val_eur > 0 else 0
                    })
                
                sector_df_full = pd.DataFrame(sector_data)
                
                # Group by industry (aggregate if multiple sectors map to same industry)
                industry_df = sector_df_full.groupby("Industry", as_index=False).agg({
                    "Value": "sum",
                    "% of Portfolio": "sum",
                    "Full_Sector": lambda x: "<br>".join(x) if len(x) > 1 else x.iloc[0]
                }).sort_values("Value", ascending=False)
                
                # Top 10 + Other
                if len(industry_df) > 10:
                    top_10 = industry_df.head(10)
                    other_value = industry_df.iloc[10:]["Value"].sum()
                    other_pct = industry_df.iloc[10:]["% of Portfolio"].sum()
                    other_sectors = "<br>".join(industry_df.iloc[10:]["Full_Sector"].tolist())
                    
                    other_row = pd.DataFrame([{
                        "Industry": "Other",
                        "Value": other_value,
                        "% of Portfolio": other_pct,
                        "Full_Sector": other_sectors
                    }])
                    industry_df = pd.concat([top_10, other_row], ignore_index=True)
                
                st.caption("Industry Allocation %")
                
                # Smart color mapping by industry category
                def get_sector_color(sector_name):
                    """Assign colors based on industry categories"""
                    sector_lower = sector_name.lower()
                    
                    # Technology - Blues
                    if any(word in sector_lower for word in ['technology', 'ai', 'machine learning', 'quantum', 'semiconductor', '3d printing', 'software']):
                        if 'quantum' in sector_lower:
                            return '#4A90E2'  # Light blue
                        elif 'ai' in sector_lower or 'machine' in sector_lower:
                            return '#357ABD'  # Medium blue
                        elif 'semiconductor' in sector_lower:
                            return '#2E5F8F'  # Darker blue
                        else:
                            return '#6BAED6'  # Sky blue
                    
                    # Industrials - Grays/Silvers
                    elif any(word in sector_lower for word in ['industrial', 'aerospace', 'defense', 'manufacturing', 'equipment', 'security']):
                        if 'aerospace' in sector_lower or 'defense' in sector_lower:
                            return '#7F8C8D'  # Dark gray
                        elif 'manufacturing' in sector_lower or 'equipment' in sector_lower:
                            return '#95A5A6'  # Medium gray
                        else:
                            return '#BDC3C7'  # Light gray
                    
                    # Healthcare - Greens
                    elif any(word in sector_lower for word in ['healthcare', 'biotech', 'genomic', 'medical', 'life sciences']):
                        if 'biotech' in sector_lower or 'genomic' in sector_lower:
                            return '#27AE60'  # Emerald
                        elif 'medical' in sector_lower:
                            return '#2ECC71'  # Green
                        else:
                            return '#52D273'  # Light green
                    
                    # Financial - Yellows/Golds
                    elif any(word in sector_lower for word in ['financial', 'fintech', 'bank']):
                        return '#F39C12'  # Orange-gold
                    
                    # Consumer/E-commerce - Oranges
                    elif any(word in sector_lower for word in ['consumer', 'e-commerce', 'ecommerce', 'digital media']):
                        return '#E67E22'  # Orange
                    
                    # Automotive/Mobility - Reds/Pinks
                    elif any(word in sector_lower for word in ['automotive', 'mobility', 'evtol', 'drone', 'autonomous', 'lidar', 'telematic']):
                        if 'evtol' in sector_lower or 'drone' in sector_lower:
                            return '#E74C3C'  # Red
                        elif 'ev' in sector_lower:
                            return '#C0392B'  # Dark red
                        else:
                            return '#EC7063'  # Light red
                    
                    # Energy - Purples
                    elif any(word in sector_lower for word in ['energy', 'clean tech', 'utilities', 'grid', 'nuclear', 'solar', 'wind']):
                        if 'nuclear' in sector_lower:
                            return '#8E44AD'  # Purple
                        elif 'solar' in sector_lower or 'wind' in sector_lower:
                            return '#9B59B6'  # Light purple
                        else:
                            return '#BB8FCE'  # Pale purple
                    
                    # Default - Teal
                    else:
                        return '#16A085'  # Teal
                
                # Create color list based on industry names
                colors = [get_sector_color(industry) for industry in industry_df['Industry']]
                
                # Pie chart with smart colors - show industry, details on hover
                fig = px.pie(industry_df, values="% of Portfolio", names="Industry",
                            title="",
                            color_discrete_sequence=colors,
                            custom_data=['Full_Sector', 'Value'])
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent',  # Only show percentage inside
                    hovertemplate='<b>%{label}</b><br>%{customdata[0]}<br>€%{customdata[1]:,.2f}<br>%{percent}<extra></extra>'
                )
                fig.update_layout(showlegend=True, height=450, legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=10)
                ))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Portfolio Value Over Time
        try:
            history = load_portfolio_history(DATA_DIR)
            if not history.empty and len(history) > 1:
                st.subheader("📈 Portfolio Value Over Time")
                history["Date"] = pd.to_datetime(history["Date"])
                st.line_chart(history.set_index("Date")["Value_EUR"], use_container_width=True)
        except:
            pass

        st.divider()

        # Watchlist Opportunities (enhanced)
        st.subheader("🎯 Watchlist Opportunities")
        wl = load_csv(WATCHLIST_CSV, WATCHLIST_COLS)
        
        if not wl.empty and MARKETSTACK_KEY:
            try:
                # Fetch watchlist prices quickly
                wl_symbols = [str(s) for s in wl["Symbol"].dropna().unique().tolist()]
                wl_prices = fetch_eod_prices(wl_symbols, MARKETSTACK_KEY) if wl_symbols else {}
                
                # Collect all opportunities with upside data
                all_opportunities = []
                for _, r in wl.iterrows():
                    try:
                        user_sym = str(r["Symbol"]) if pd.notna(r["Symbol"]) else ""
                        if not user_sym:
                            continue
                            
                        name = str(r.get("Name", user_sym) or user_sym)
                        user_ccy = str(r.get("Currency", "EUR") or "EUR").upper()
                        
                        # Get numeric values
                        buy_low = float(r.get("Buy_Low", 0) or 0)
                        buy_high = float(r.get("Buy_High", 0) or 0)
                        target_1y = float(r.get("Target_1Y", 0) or 0)
                        target_3y = float(r.get("Target_3Y", 0) or 0)
                        target_5y = float(r.get("Target_5Y", 0) or 0)
                        target_10y = float(r.get("Target_10Y", 0) or 0)
                        
                        # Get current price
                        current_price = wl_prices.get(user_sym)
                        if not current_price:
                            continue
                        
                        current_eur = to_eur(current_price, user_ccy, FX_MAP)
                        if not current_eur or current_eur <= 0:
                            continue
                        
                        # Convert targets to EUR
                        buy_low_eur = to_eur(buy_low, user_ccy, FX_MAP) if buy_low else 0
                        buy_high_eur = to_eur(buy_high, user_ccy, FX_MAP) if buy_high else 0
                        target_1y_eur = to_eur(target_1y, user_ccy, FX_MAP) if target_1y else 0
                        target_3y_eur = to_eur(target_3y, user_ccy, FX_MAP) if target_3y else 0
                        target_5y_eur = to_eur(target_5y, user_ccy, FX_MAP) if target_5y else 0
                        target_10y_eur = to_eur(target_10y, user_ccy, FX_MAP) if target_10y else 0
                        
                        # Calculate upsides
                        upside_1y = ((target_1y_eur - current_eur) / current_eur * 100) if target_1y_eur > 0 else 0
                        upside_3y = ((target_3y_eur - current_eur) / current_eur * 100) if target_3y_eur > 0 else 0
                        upside_5y = ((target_5y_eur - current_eur) / current_eur * 100) if target_5y_eur > 0 else 0
                        upside_10y = ((target_10y_eur - current_eur) / current_eur * 100) if target_10y_eur > 0 else 0
                        
                        # Distance from buy low (negative means below target, positive means above)
                        distance_from_low = ((current_eur - buy_low_eur) / buy_low_eur * 100) if buy_low_eur > 0 else 999
                        
                        all_opportunities.append({
                            "name": name[:30],
                            "symbol": user_sym,
                            "current": current_eur,
                            "buy_low": buy_low_eur,
                            "buy_high": buy_high_eur,
                            "upside_1y": upside_1y,
                            "upside_3y": upside_3y,
                            "upside_5y": upside_5y,
                            "upside_10y": upside_10y,
                            "distance_from_low": distance_from_low
                        })
                    except Exception:
                        continue
                
                if all_opportunities:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🚀 Top Upside Potential by Timeframe**")
                        
                        # Get top 3 unique stocks per timeframe
                        seen_symbols = set()
                        top_upside = []
                        
                        for timeframe, key in [("1Y", "upside_1y"), ("3Y", "upside_3y"), ("5Y", "upside_5y"), ("10Y", "upside_10y")]:
                            sorted_opps = sorted([o for o in all_opportunities if o[key] > 0], key=lambda x: x[key], reverse=True)
                            count = 0
                            for opp in sorted_opps:
                                if opp["symbol"] not in seen_symbols and count < 3:
                                    top_upside.append({
                                        "Stock": opp["name"],
                                        "Timeframe": timeframe,
                                        "Current (€)": f"€{opp['current']:.2f}",
                                        "Upside %": f"+{opp[key]:.1f}%"
                                    })
                                    seen_symbols.add(opp["symbol"])
                                    count += 1
                                if count >= 3:
                                    break
                        
                        if top_upside:
                            upside_df = pd.DataFrame(top_upside)
                            st.dataframe(upside_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No upside targets available")
                    
                    with col2:
                        st.markdown("**💰 Best Entry Opportunities (Near Buy Low)**")
                        
                        # Get stocks closest to (or below) buy low target
                        buy_opps = [o for o in all_opportunities if o["buy_low"] > 0]
                        buy_opps_sorted = sorted(buy_opps, key=lambda x: x["distance_from_low"])[:10]
                        
                        if buy_opps_sorted:
                            buy_df = pd.DataFrame([{
                                "Stock": o["name"],
                                "Current (€)": f"€{o['current']:.2f}",
                                "Target (€)": f"€{o['buy_low']:.2f}",
                                "vs Target": f"{o['distance_from_low']:+.1f}%"
                            } for o in buy_opps_sorted])
                            st.dataframe(buy_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No buy targets available")
                else:
                    st.info("No opportunities found. Add targets to your watchlist.")
            except Exception as e:
                st.warning(f"Could not load watchlist opportunities: {str(e)}")
        else:
            st.info("Add stocks to your Watchlist to see opportunities.")

# ----------------------------
# Portfolio
# ----------------------------
with tabs[1]:
    st.subheader("Portfolio (EUR view, EOD, auto FX, alt listing fallback)")
    st.caption("Columns: Name, Symbol, Currency, Quantity, BEP, Sector, Industry, Multiplier (optional; e.g., ADR ratio)")

    templ = pd.DataFrame(columns=PORTFOLIO_COLS)
    st.download_button("Download CSV template", templ.to_csv(index=False, sep=';').encode("utf-8"),
                       "portfolio_template.csv", "text/csv")

    up = st.file_uploader("Import CSV", type=["csv"], key="port_upl")
    
    # Check if we have a pending upload for column mapping
    if st.session_state.pending_upload is not None and st.session_state.upload_type == "portfolio":
        # Show column mapping UI
        mapped_df = show_column_mapping_ui(
            st.session_state.pending_upload, 
            PORTFOLIO_COLS, 
            csv_type="portfolio",
            csv_format_info=st.session_state.csv_format_info
        )
        
        if mapped_df is not None:
            try:
                # Clean up any potential parsing issues
                for col in ['Quantity', 'BEP']:
                    if col in mapped_df.columns:
                        mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce').fillna(0.0)
                
                save_csv(PORTFOLIO_CSV, mapped_df)
                
                # Clear pending upload BEFORE showing success
                st.session_state.pending_upload = None
                st.session_state.upload_type = None
                st.session_state.csv_format_info = None
                
                st.success(f"✅ Portfolio imported successfully! {len(mapped_df)} rows loaded.")
                st.rerun()
                    
            except Exception as e:
                st.error(f"Error saving mapped CSV: {str(e)}")
    
    elif up:
        try:
            # Use smart CSV reader to handle both European and standard formats
            df, delimiter, decimal_sep = read_csv_smart(up)
            
            # Store for mapping
            st.session_state.pending_upload = df.copy()
            st.session_state.upload_type = "portfolio"
            st.session_state.csv_format_info = (delimiter, decimal_sep)
            st.rerun()
                
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            st.info("Please ensure your CSV is properly formatted.")
    
    # Only show portfolio table and controls when NOT in mapping mode
    if st.session_state.pending_upload is None or st.session_state.upload_type != "portfolio":
        port = load_csv(PORTFOLIO_CSV, PORTFOLIO_COLS)

        with st.expander("Add / Edit Row"):
            st.caption("💡 Type a company name or ticker symbol - autocomplete will help fill the rest!")
            
            # Search input
            search_col1, search_col2 = st.columns([3, 1])
            
            with search_col1:
                search_query = st.text_input(
                    "Search Company", 
                    placeholder="Type company name or symbol (e.g., 'Apple' or 'AAPL')",
                    key="company_search",
                    help="Start typing to search for stocks"
                )
            
            with search_col2:
                search_btn = st.button("🔍 Search", type="secondary", use_container_width=True)
            
            # Initialize session state for selected company
            if 'selected_company' not in st.session_state:
                st.session_state.selected_company = None
            
            # Perform search
            if search_btn and search_query:
                with st.spinner("Searching..."):
                    results = search_companies(search_query, MARKETSTACK_KEY, limit=10)
                    if results:
                        st.session_state.search_results = results
                    else:
                        st.warning("No results found. Try a different search term.")
                        st.session_state.search_results = []
            
            # Show search results
            if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
                st.write("**Select a company:**")
                for idx, company in enumerate(st.session_state.search_results):
                    col_name, col_btn = st.columns([4, 1])
                    with col_name:
                        sector_text = f" • {company['sector']}" if company['sector'] else " • Sector unknown"
                        st.write(f"**{company['name']}** ({company['symbol']}) - {company['currency']}{sector_text}")
                    with col_btn:
                        if st.button("Select", key=f"select_{idx}", use_container_width=True):
                            st.session_state.selected_company = company
                            st.session_state.search_results = []
                            st.rerun()
                
                st.divider()
            
            # Form with pre-filled or manual entry
            with st.form("add_port"):
                # Get available sectors
                available_sectors = get_sector_list()
                
                # Pre-fill from selected company or allow manual entry
                default_name = st.session_state.selected_company['name'] if st.session_state.selected_company else ""
                default_symbol = st.session_state.selected_company['symbol'] if st.session_state.selected_company else ""
                default_sector = st.session_state.selected_company['sector'] if st.session_state.selected_company and st.session_state.selected_company['sector'] else None
                default_currency = st.session_state.selected_company['currency'] if st.session_state.selected_company else "EUR"
                
                c1,c2 = st.columns(2)
                name = c1.text_input("Name*", value=default_name, help="Company name (e.g., NVIDIA Corporation)")
                symbol = c2.text_input("Symbol*", value=default_symbol, help="Stock ticker (e.g., NVDA)")
                
                c3,c4 = st.columns(2)
                
                # Sector dropdown with predefined list
                if default_sector and default_sector in available_sectors:
                    sector_index = available_sectors.index(default_sector)
                else:
                    sector_index = 0
                
                sector = c3.selectbox(
                    "Sector*", 
                    options=available_sectors,
                    index=sector_index,
                    help="Select from predefined sectors"
                )
                
                currency_options = ["USD","EUR","GBP","CHF"]
                currency_index = currency_options.index(default_currency) if default_currency in currency_options else currency_options.index("EUR")
                currency = c4.selectbox("Currency*", currency_options, index=currency_index, help="Stock's native currency")
                
                c5,c6 = st.columns(2)
                qty = c5.number_input("Quantity*", min_value=0.0, step=1.0, help="Number of shares")
                bep = c6.number_input("BEP*", min_value=0.0, step=0.01, format="%.2f", help="Break-even price per share in native currency")
                
                st.caption("Fields marked with * are required. ISIN, Industry, Country will be fetched from API. Current price, Value, and P/L are calculated automatically.")
                
                col_save, col_clear = st.columns(2)
                with col_save:
                    add_btn = st.form_submit_button("💾 Save Position", type="primary", use_container_width=True)
                with col_clear:
                    clear_btn = st.form_submit_button("🔄 Clear Form", use_container_width=True)
            
            if clear_btn:
                st.session_state.selected_company = None
                if hasattr(st.session_state, 'search_results'):
                    st.session_state.search_results = []
                st.rerun()
            
            if add_btn:
                if not name or not symbol or not sector or not qty or not bep:
                    st.error("Please fill in all required fields marked with *")
                else:
                    new = pd.DataFrame([{
                        "Name": name,
                        "Symbol": symbol,
                        "Quantity": qty,
                        "BEP": bep,
                        "Sector": sector,
                        "Currency": currency
                    }])
                    port = pd.concat([port, new], ignore_index=True)
                    save_csv(PORTFOLIO_CSV, port)
                    st.success("Position added! Refresh to see updated values.")
                    st.rerun()

        # Action buttons
        col_refresh, col_clear, col_info = st.columns([1, 1, 3])
        with col_refresh:
            if st.button("🔄 Refresh Prices", help="Force refresh current prices (ignore today's cache and fetch fresh Marketstack data)"):
                # Clear in-memory and on-disk caches so next fetch will retrieve fresh data
                st.session_state.prices_cache.clear()
                st.session_state.last_price_fetch = None
                # Also clear FX cache to force fresh FX fetch when needed
                st.session_state.fx_cache = {}
                st.session_state.last_fx_fetch = None
                # Remove persisted cache files
                try:
                    if PRICES_CACHE_FILE.exists(): PRICES_CACHE_FILE.unlink()
                except Exception:
                    pass
                try:
                    if FX_CACHE_FILE.exists(): FX_CACHE_FILE.unlink()
                except Exception:
                    pass
                st.rerun()
        
        with col_clear:
            if st.button("🗑️ Clear Portfolio", help="Remove all portfolio data", type="secondary"):
                # Clear the CSV file
                empty_df = pd.DataFrame(columns=PORTFOLIO_COLS)
                save_csv(PORTFOLIO_CSV, empty_df)
                # Clear session cache
                st.session_state.prices_cache.clear()
                st.session_state.last_price_fetch = None
                st.success("Portfolio cleared!")
                st.rerun()
        
        with col_info:
            cache_age = ""
            if st.session_state.last_price_fetch:
                age_minutes = (datetime.now() - st.session_state.last_price_fetch).total_seconds() / 60
                cache_age = f"(Prices cached {age_minutes:.1f} min ago)"
            st.caption(f"Portfolio values in EUR {cache_age}")

        # Auto-refresh prices on page load (if cache is older than 5 minutes)
        auto_refresh = False
        if st.session_state.last_price_fetch:
            age_seconds = (datetime.now() - st.session_state.last_price_fetch).total_seconds()
            if age_seconds > 300:  # 5 minutes
                auto_refresh = True
        else:
            auto_refresh = True  # First load, always fetch
        
        if auto_refresh:
            st.session_state.last_price_fetch = None  # Clear cache to force refresh

        # Pricing block - FAST batch processing
        with st.spinner("Loading current prices..."):
            # First pass: collect all provider symbols using ONLY overrides (no API search)
            all_provider_symbols = set()
            symbol_map = {}  # user_sym -> (provider_sym, ccy)
            overrides = load_overrides()
            
            for _, row in port.iterrows():
                user_sym = str(row["Symbol"]) if pd.notna(row["Symbol"]) else ""
                name_hint = str(row["Name"] or "") if pd.notna(row["Name"]) else ""
                user_ccy = str(row["Currency"] or "EUR").upper() if pd.notna(row["Currency"]) else "EUR"
                
                if user_sym:
                    # Check override first
                    ov = overrides[overrides["UserSymbol"].fillna("").str.upper() == user_sym.upper()]
                    if not ov.empty:
                        psym = ov.iloc[0].get("ProviderSymbol", user_sym).strip()
                        pccy = ov.iloc[0].get("ProviderCurrency", user_ccy).strip()
                    else:
                        # No override: use direct symbol
                        psym = user_sym
                        pccy = user_ccy
                    
                    symbol_map[user_sym] = (psym, pccy)
                    all_provider_symbols.add(psym)
            
            # Fetch ALL prices in ONE batch call
            if all_provider_symbols and MARKETSTACK_KEY:
                prices = fetch_eod_prices(list(all_provider_symbols), MARKETSTACK_KEY)
            else:
                prices = {}
            
            # Check for missing prices and try ONE search per missing symbol
            missing_symbols = [(us, pm, pc) for us, (pm, pc) in symbol_map.items() if prices.get(pm) is None]
            
            if missing_symbols and len(missing_symbols) <= 10:  # Only auto-resolve if < 10 missing
                resolved_count = 0
                for user_sym, provider_sym, user_ccy in missing_symbols:
                    # Get name for better search
                    row_data = port[port["Symbol"] == user_sym]
                    if not row_data.empty:
                        name_hint = str(row_data.iloc[0]["Name"] or "")
                        
                        # Try ONE search by name
                        if name_hint:
                            cand = search_alternative_symbols(name_hint, MARKETSTACK_KEY, limit=3)
                            best = pick_best_alt(cand, name_hint)
                            if best and best.get("symbol"):
                                new_sym = best["symbol"]
                                new_ccy = best.get("currency", user_ccy)
                                
                                # Fetch price for this one symbol
                                new_price = fetch_eod_prices([new_sym], MARKETSTACK_KEY)
                                if new_price.get(new_sym):
                                    prices[new_sym] = new_price[new_sym]
                                    symbol_map[user_sym] = (new_sym, new_ccy)
                                    
                                    # Save override
                                    save_override(user_sym, new_sym, new_ccy)
                                    resolved_count += 1
                
                if resolved_count > 0:
                    st.success(f"✓ Auto-resolved {resolved_count} missing symbols")

        def calc_row(_r):
            qty = float(_r["Quantity"] or 0)
            user_sym = str(_r["Symbol"]) if pd.notna(_r["Symbol"]) else ""
            user_ccy = str(_r["Currency"] or "EUR").upper() if pd.notna(_r["Currency"]) else "EUR"
            bep = float(_r["BEP"] or 0.0)
            
            # Use pre-mapped symbol
            psym, pccy = symbol_map.get(user_sym, (user_sym, user_ccy))

            # Now we should have all prices already
            last = prices.get(psym, None)
            last_eur = to_eur(last, pccy, FX_MAP) if last is not None else None

            bep_eur = to_eur(bep, user_ccy, FX_MAP) if bep else None
            val_eur = qty * last_eur if (qty and last_eur is not None) else None
            cost_eur = qty * bep_eur if (qty and bep_eur is not None) else None
            pnl = (val_eur - cost_eur) if (val_eur is not None and cost_eur is not None) else None
            pnl_pct = (pnl / cost_eur * 100) if (pnl is not None and cost_eur and cost_eur!=0) else None
            return last_eur, val_eur, pnl, pnl_pct, psym, pccy

        # Only calculate prices if we have data and API key
        if not port.empty and MARKETSTACK_KEY:
            view = port.copy()
            last_eurs, vals, pnls, pnl_pcts, provs, pccys = [], [], [], [], [], []
            
            # Show progress for slow operations
            with st.spinner("Fetching current prices..."):
                for _, r in view.iterrows():
                    le, v, p, pp, ps, pc = calc_row(r)
                    last_eurs.append(le); vals.append(v); pnls.append(p); pnl_pcts.append(pp)
                    provs.append(ps); pccys.append(pc)

            # Create a clean display dataframe with numeric columns for proper sorting
            display_df = pd.DataFrame({
                "Name": view["Name"],
                "Symbol": view["Symbol"],
                "Sector": view["Sector"],
                "Currency": view["Currency"],
                "Quantity": view["Quantity"],
                "BEP": view["BEP"],
                "Current Price (EUR)": last_eurs,  # Keep as numeric for sorting
                "Value (EUR)": vals,  # Keep as numeric, show full value
                "P/L (EUR)": pnls,  # Keep as numeric for sorting
                "P/L %": pnl_pcts  # Keep as numeric for sorting
            })
            
            # Save the updated values back to the original structure
            view["Value"] = vals
            view["Total P/L€"] = pnls  
            view["Total P/L%"] = pnl_pcts
        else:
            display_df = port.copy()
            if not MARKETSTACK_KEY:
                st.warning("⚠️ Marketstack API key missing. Add it in the sidebar to fetch current prices.")
            provs, pccys, mults, last_eurs = [], [], [], []
        
        # Create editable version - allow editing core fields
        editable_df = display_df.copy()
        
        # Make calculated columns read-only by moving them after editing
        editable_columns = ["Name", "Symbol", "Quantity", "BEP", "Sector", "Currency"]
        readonly_columns = ["Current Price (EUR)", "Value (EUR)", "P/L (EUR)", "P/L %"]
        
        # Reorder for better UX: editable first, then readonly
        column_order = editable_columns + readonly_columns
        editable_df = editable_df[column_order]
        
        st.write("**📊 Portfolio Holdings** - Click column headers to sort")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Quantity": st.column_config.NumberColumn(
                    "Quantity",
                    format="%.0f",
                    width="small"
                ),
                "BEP": st.column_config.NumberColumn(
                    "BEP",
                    format="%.2f",
                    help="Break-even price in original currency",
                    width="small"
                ),
                "Sector": st.column_config.TextColumn(
                    "Sector",
                    help="Stock sector/category",
                    width="large"
                ),
                "Currency": st.column_config.SelectboxColumn(
                    "Currency",
                    options=["EUR", "USD", "GBP", "CHF", "CAD", "SEK", "NOK", "DKK", "JPY", "HKD", "AUD", "PLN"],
                    width="small"
                ),
                "Current Price (EUR)": st.column_config.NumberColumn(
                    "Current Price (EUR)",
                    format="€%.2f",
                    disabled=True,  # Read-only
                    width="small"
                ),
                "Value (EUR)": st.column_config.NumberColumn(
                    "Value (EUR)",
                    format="€%.2f",
                    disabled=True,  # Read-only
                    width="medium"
                ),
                "P/L (EUR)": st.column_config.NumberColumn(
                    "P/L (EUR)",
                    format="€%.2f",
                    disabled=True,  # Read-only
                    width="medium"
                ),
                "P/L %": st.column_config.NumberColumn(
                    "P/L %",
                    format="%.1f%%",
                    disabled=True,  # Read-only
                    width="small"
                )
            },
            key="portfolio_editor"
        )
        st.caption('Use Add/Edit form above - Click headers to sort')

        # Diagnostic info
        
        # Add diagnostic columns for debugging (can be hidden later)
        if not port.empty and MARKETSTACK_KEY:
            with st.expander("🔧 Advanced Info (Provider Data)"):
                debug_view = port.copy()
                if provs:  # Only add if we have data
                    debug_view["ProviderSymbol"] = provs
                    debug_view["ProviderCcy"] = pccys
                st.dataframe(debug_view, use_container_width=True)

# ----------------------------
# Watchlists
# ----------------------------
with tabs[2]:
    st.subheader("Watchlists (targets & distance, EUR)")
    st.caption("Columns: List, Name, Symbol, Currency, BuyLow, BuyHigh, Sector, Industry, Target1Y, Target3Y, Target5Y, Target10Y, Multiplier")

    templw = pd.DataFrame(columns=WATCHLIST_COLS)
    st.download_button("Download CSV template", templw.to_csv(index=False, sep=';').encode("utf-8"),
                       "watchlist_template.csv", "text/csv")

    upw = st.file_uploader("Import CSV", type=["csv"], key="wl_upl")
    
    # Check if we have a pending upload for column mapping
    if st.session_state.pending_upload is not None and st.session_state.upload_type == "watchlist":
        mapped_dfw = show_column_mapping_ui(
            st.session_state.pending_upload, 
            WATCHLIST_COLS, 
            csv_type="watchlist",
            csv_format_info=st.session_state.csv_format_info
        )
        
        if mapped_dfw is not None:
            try:
                # Clean up numeric columns
                for col in ['Buy_Low', 'Buy_High', 'Target_1Y']:
                    if col in mapped_dfw.columns:
                        mapped_dfw[col] = pd.to_numeric(mapped_dfw[col], errors='coerce')
                
                save_csv(WATCHLIST_CSV, mapped_dfw)
                
                # Clear pending upload BEFORE showing success
                st.session_state.pending_upload = None
                st.session_state.upload_type = None
                st.session_state.csv_format_info = None
                
                st.success(f"✅ Watchlist imported successfully! {len(mapped_dfw)} rows loaded.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error saving mapped watchlist CSV: {str(e)}")
    
    elif upw:
        try:
            # Use smart CSV reader to handle both European and standard formats
            dfw, delimiter, decimal_sep = read_csv_smart(upw)
            
            # Store for mapping
            st.session_state.pending_upload = dfw.copy()
            st.session_state.upload_type = "watchlist"
            st.session_state.csv_format_info = (delimiter, decimal_sep)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error reading watchlist CSV: {str(e)}")
            st.info("Please ensure your CSV is properly formatted.")

    # Only show watchlist table and controls when NOT in mapping mode
    if st.session_state.pending_upload is None or st.session_state.upload_type != "watchlist":
        wl = load_csv(WATCHLIST_CSV, WATCHLIST_COLS)

        with st.expander("Add / Edit Row"):
            st.caption("💡 Type a company name or ticker symbol - autocomplete will help fill the rest!")
            
            # Search input
            search_col1, search_col2 = st.columns([3, 1])
            
            with search_col1:
                search_query_wl = st.text_input(
                    "Search Company", 
                    placeholder="Type company name or symbol (e.g., 'Apple' or 'AAPL')",
                    key="company_search_wl",
                    help="Start typing to search for stocks"
                )
            
            with search_col2:
                search_btn_wl = st.button("🔍 Search", type="secondary", use_container_width=True, key="search_btn_wl")
            
            # Initialize session state for selected company
            if 'selected_company_wl' not in st.session_state:
                st.session_state.selected_company_wl = None
            
            # Perform search
            if search_btn_wl and search_query_wl:
                with st.spinner("Searching..."):
                    results = search_companies(search_query_wl, MARKETSTACK_KEY, limit=10)
                    if results:
                        st.session_state.search_results_wl = results
                    else:
                        st.warning("No results found. Try a different search term.")
                        st.session_state.search_results_wl = []
            
            # Show search results
            if hasattr(st.session_state, 'search_results_wl') and st.session_state.search_results_wl:
                st.write("**Select a company:**")
                for idx, company in enumerate(st.session_state.search_results_wl):
                    col_name, col_btn = st.columns([4, 1])
                    with col_name:
                        sector_text = f" • {company['sector']}" if company['sector'] else " • Sector unknown"
                        st.write(f"**{company['name']}** ({company['symbol']}) - {company['currency']}{sector_text}")
                    with col_btn:
                        if st.button("Select", key=f"select_wl_{idx}", use_container_width=True):
                            st.session_state.selected_company_wl = company
                            st.session_state.search_results_wl = []
                            st.rerun()
                
                st.divider()
            
            # Form with pre-filled or manual entry
            with st.form("add_wl"):
                # Get available sectors
                available_sectors = get_sector_list()
                
                # Pre-fill from selected company or allow manual entry
                default_name_wl = st.session_state.selected_company_wl['name'] if st.session_state.selected_company_wl else ""
                default_symbol_wl = st.session_state.selected_company_wl['symbol'] if st.session_state.selected_company_wl else ""
                default_sector_wl = st.session_state.selected_company_wl['sector'] if st.session_state.selected_company_wl and st.session_state.selected_company_wl['sector'] else None
                default_currency_wl = st.session_state.selected_company_wl['currency'] if st.session_state.selected_company_wl else "EUR"
                
                c1,c2 = st.columns(2)
                wname = c1.text_input("Name*", value=default_name_wl, help="Company name")
                wsym = c2.text_input("Symbol*", value=default_symbol_wl, help="Stock ticker (e.g., NVDA)")
                
                c3,c4 = st.columns(2)
                
                # Sector dropdown with predefined list
                if default_sector_wl and default_sector_wl in available_sectors:
                    sector_index_wl = available_sectors.index(default_sector_wl)
                else:
                    sector_index_wl = 0
                
                wsec = c3.selectbox(
                    "Sector*", 
                    options=available_sectors,
                    index=sector_index_wl,
                    help="Select from predefined sectors",
                    key="watchlist_sector"
                )
                
                currency_options = ["USD","EUR","GBP","CHF"]
                currency_index_wl = currency_options.index(default_currency_wl) if default_currency_wl in currency_options else currency_options.index("EUR")
                wccy = c4.selectbox("Currency*", currency_options, index=currency_index_wl, help="Stock's native currency", key="watchlist_currency")
                
                st.write("**Optional Manual Overrides** (leave blank for auto-calculation)")
                d1,d2,d3 = st.columns(3)
                buyl = d1.number_input("Buy Low", min_value=0.0, step=0.01, format="%.2f", help="Leave 0 for auto: current price × 0.9")
                buyh = d2.number_input("Buy High", min_value=0.0, step=0.01, format="%.2f", help="Leave 0 for auto: current price × 1.1")
                t1y = d3.number_input("Target 1Y", min_value=0.0, step=0.01, format="%.2f", help="Leave 0 for auto: price × (1 + sector growth)")
                
                st.caption("Fields marked with * are required. Buy Low/High auto = current price ±10%. Target 1Y = price × sector growth. Targets 3Y/5Y/10Y auto-calculated from 1Y.")
                
                col_save_wl, col_clear_wl = st.columns(2)
                with col_save_wl:
                    addw = st.form_submit_button("💾 Save to Watchlist", type="primary", use_container_width=True)
                with col_clear_wl:
                    clear_btn_wl = st.form_submit_button("🔄 Clear Form", use_container_width=True)
            
            if clear_btn_wl:
                st.session_state.selected_company_wl = None
                if hasattr(st.session_state, 'search_results_wl'):
                    st.session_state.search_results_wl = []
                st.rerun()
            
            if addw:
                if not wname or not wsym or not wsec:
                    st.error("Please fill in all required fields marked with *")
                else:
                    neww = pd.DataFrame([{
                        "Name": wname,
                        "Symbol": wsym,
                        "Sector": wsec,
                        "Currency": wccy,
                        "Buy_Low": buyl if buyl > 0 else None,
                        "Buy_High": buyh if buyh > 0 else None,
                        "Target_1Y": t1y if t1y > 0 else None,
                        "Target_3Y": None,  # Auto-calculated
                        "Target_5Y": None,  # Auto-calculated
                        "Target_10Y": None  # Auto-calculated
                    }])
                    wl = pd.concat([wl, neww], ignore_index=True)
                    save_csv(WATCHLIST_CSV, wl)
                    st.success("Added to watchlist! Refresh to see auto-calculated values.")
                    st.rerun()

        # Action buttons
        col_refresh_wl, col_clear_wl, col_info_wl = st.columns([1, 1, 3])
        with col_refresh_wl:
            if st.button("🔄 Refresh Prices", help="Force refresh current prices (ignore today's cache and fetch fresh Marketstack data)", key="refresh_wl"):
                # Clear only the relevant cache, then clear persisted cache files so next load fetches fresh data
                if 'prices_cache' in st.session_state:
                    st.session_state.prices_cache.clear()
                if 'last_price_fetch' in st.session_state:
                    st.session_state.last_price_fetch = None
                if 'fx_cache' in st.session_state:
                    st.session_state.fx_cache = {}
                if 'last_fx_fetch' in st.session_state:
                    st.session_state.last_fx_fetch = None
                try:
                    if PRICES_CACHE_FILE.exists(): PRICES_CACHE_FILE.unlink()
                except Exception:
                    pass
                try:
                    if FX_CACHE_FILE.exists(): FX_CACHE_FILE.unlink()
                except Exception:
                    pass
                st.rerun()
        
        with col_clear_wl:
            if st.button("🗑️ Clear Watchlist", help="Remove all watchlist data", type="secondary", key="clear_watchlist"):
                # Clear the CSV file
                empty_df = pd.DataFrame(columns=WATCHLIST_COLS)
                save_csv(WATCHLIST_CSV, empty_df)
                st.success("Watchlist cleared!")
                st.rerun()
        
        with col_info_wl:
            cache_age = ""
            if st.session_state.last_price_fetch:
                age_minutes = (datetime.now() - st.session_state.last_price_fetch).total_seconds() / 60
                cache_age = f" (cached {age_minutes:.1f} min ago)"
            st.caption(f"💡 Prices auto-refresh every 5 minutes{cache_age}")

        # First resolve all user symbols to provider symbols using overrides
        all_provider_symbols_w = set()
        prices_w = {}  # Start with empty price dict
        
        for _, row in wl.iterrows():
            user_sym = str(row["Symbol"]) if pd.notna(row["Symbol"]) else ""
            name_hint = str(row["Name"] or "") if pd.notna(row["Name"]) else ""
            row_ccy = str(row.get("Currency", "EUR") or "EUR").strip() if pd.notna(row.get("Currency")) else "EUR"
            if user_sym:
                # Resolve using overrides FIRST (with empty prices_w initially)
                psym, _ = resolve_provider_symbol(user_sym, name_hint, row_ccy, MARKETSTACK_KEY, prices_w)
                if psym:
                    all_provider_symbols_w.add(psym)
        
        # Now fetch prices for all resolved provider symbols in one batch
        if all_provider_symbols_w and MARKETSTACK_KEY:
            prices_w = fetch_eod_prices(list(all_provider_symbols_w), MARKETSTACK_KEY)
        
        # Fetch 52-week high/low data for all provider symbols
        week52_data = fetch_52week_data(list(all_provider_symbols_w), MARKETSTACK_KEY) if MARKETSTACK_KEY and all_provider_symbols_w else {}

        def wl_calc_row(r):
            user_sym = str(r["Symbol"]) if pd.notna(r["Symbol"]) else ""
            name_hint = str(r["Name"] or "") if pd.notna(r["Name"]) else ""
            sector = str(r.get("Sector") or "").strip()
            row_ccy = str(r.get("Currency", "EUR") or "EUR").strip() if pd.notna(row.get("Currency")) else "EUR"
            
            psym, pccy = resolve_provider_symbol(user_sym, name_hint, row_ccy, MARKETSTACK_KEY, prices_w)

            # Now we should have all prices already - no individual API calls needed
            last = prices_w.get(psym, None)
            last_eur = to_eur(last, pccy, FX_MAP) if last is not None else None
            
            # 52-week data not fetched automatically for performance
            week52_high = None
            week52_low = None

            def fnum(x):
                return float(x) if x is not None and not pd.isna(x) else None

            # Get sector growth rate
            growth_rate = None
            if sector:
                growth_df = load_csv(IND_GROWTH_CSV, ["Industry","Sector","YoY_Growth_%"])
                sector_rows = growth_df[growth_df["Sector"].str.upper() == sector.upper()]
                if not sector_rows.empty:
                    growth_rate = sector_rows["YoY_Growth_%"].mean() / 100  # Convert to decimal

            # Get manual values or auto-populate
            buyl = fnum(r.get("Buy_Low"))
            buyh = fnum(r.get("Buy_High"))
            t1  = fnum(r.get("Target_1Y"))
            t3  = fnum(r.get("Target_3Y"))
            t5  = fnum(r.get("Target_5Y"))
            t10 = fnum(r.get("Target_10Y"))
            
            # Auto-populate Buy Low/High with current price if not set
            if buyl is None and last_eur is not None:
                buyl = last_eur * 0.9  # Default: 10% below current price
            
            if buyh is None and last_eur is not None:
                buyh = last_eur * 1.1  # Default: 10% above current price
            
            # Auto-populate Target 1Y if not set
            if t1 is None and last_eur is not None and growth_rate is not None:
                t1 = last_eur * (1 + growth_rate)
            
            # Calculate multi-year targets from Target 1Y and sector growth
            if t1 and growth_rate is not None:
                if t3 is None:
                    t3 = t1 * ((1 + growth_rate) ** 2)  # Compound 2 more years after year 1
                if t5 is None:
                    t5 = t1 * ((1 + growth_rate) ** 4)  # Compound 4 more years after year 1
                if t10 is None:
                    t10 = t1 * ((1 + growth_rate) ** 9)  # Compound 9 more years after year 1

            # Calculate percentage differences
            from_low = ((last_eur - buyl) / buyl * 100) if (last_eur is not None and buyl and buyl>0) else None
            from_high = ((last_eur - buyh) / buyh * 100) if (last_eur is not None and buyh and buyh>0) else None
            to_1y = ((t1 - last_eur) / last_eur * 100) if (t1 and last_eur and last_eur>0) else None
            to_3y = ((t3 - last_eur) / last_eur * 100) if (t3 and last_eur and last_eur>0) else None
            to_5y = ((t5 - last_eur) / last_eur * 100) if (t5 and last_eur and last_eur>0) else None
            to_10y = ((t10 - last_eur) / last_eur * 100) if (t10 and last_eur and last_eur>0) else None

            return (last_eur, from_low, from_high, to_1y, to_3y, to_5y, to_10y, 
                   buyl, buyh, t1, t3, t5, t10, psym, pccy)

        wv = wl.copy()
        (last_eur_list, d_low, d_high, d1, d3, d5, d10, 
         buy_lows, buy_highs, target_1ys, target_3ys, target_5ys, target_10ys, 
         ps, pc) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
        
        for _, r in wv.iterrows():
            (le, fl, fh, t1_pct, t3_pct, t5_pct, t10_pct, 
             bl, bh, t1, t3, t5, t10, pps, ppc) = wl_calc_row(r)
            last_eur_list.append(le); d_low.append(fl); d_high.append(fh); 
            d1.append(t1_pct); d3.append(t3_pct); d5.append(t5_pct); d10.append(t10_pct)
            buy_lows.append(bl); buy_highs.append(bh); 
            target_1ys.append(t1); target_3ys.append(t3); target_5ys.append(t5); target_10ys.append(t10)
            ps.append(pps); pc.append(ppc)

        # Update columns with numeric values for proper sorting
        wv["Current Price"] = last_eur_list
        wv["Buy Low"] = buy_lows
        wv["Buy High"] = buy_highs
        wv["Target 1Y"] = target_1ys
        wv["Target 3Y"] = target_3ys
        wv["Target 5Y"] = target_5ys
        wv["Target 10Y"] = target_10ys
        wv["% from Buy Low"] = d_low
        wv["% from Buy High"] = d_high
        wv["Upside to 1Y"] = d1
        wv["Upside to 3Y"] = d3
        wv["Upside to 5Y"] = d5
        wv["Upside to 10Y"] = d10

        # Add debug info for watchlist too
        with st.expander("🔧 Watchlist Advanced Info"):
            debug_wv = wv.copy()
            debug_wv["ProviderSymbol"] = ps
            debug_wv["ProviderCcy"] = pc
            st.dataframe(debug_wv, use_container_width=True)

        # Filtering controls - Compact version in expander
        with st.expander("🔍 Filter & Search Options", expanded=False):
            # Row 1: Search and quick filters
            search_col, currency_col = st.columns([2, 1])
            
            with search_col:
                search_text = st.text_input(
                    "🔎 Search",
                    placeholder="Type company name or symbol...",
                    help="Search for stocks by name or symbol",
                    label_visibility="collapsed"
                )
            
            with currency_col:
                # Get unique currencies
                all_currencies = sorted(wv["Currency"].unique().tolist()) if "Currency" in wv.columns else []
                selected_currencies = st.multiselect(
                    "💱 Currency",
                    options=all_currencies,
                    default=all_currencies,
                    help="Filter by currency",
                    label_visibility="collapsed"
                )
            
            # Row 2: Sector filter (compact multi-select)
            st.write("**Sectors:**")
            all_sectors = sorted(wv["Sector"].unique().tolist())
            selected_sectors = st.multiselect(
                "Sectors",
                options=all_sectors,
                default=all_sectors,
                help="Select sectors to display",
                label_visibility="collapsed"
            )
        
        # Apply filters
        filtered_wv = wv.copy()
        
        # Filter by sector
        if selected_sectors:
            filtered_wv = filtered_wv[filtered_wv["Sector"].isin(selected_sectors)]
        
        # Filter by currency
        if selected_currencies and "Currency" in filtered_wv.columns:
            filtered_wv = filtered_wv[filtered_wv["Currency"].isin(selected_currencies)]
        
        # Filter by search text
        if search_text:
            search_text_lower = search_text.lower()
            name_match = filtered_wv["Name"].str.lower().str.contains(search_text_lower, na=False)
            symbol_match = filtered_wv["Symbol"].str.lower().str.contains(search_text_lower, na=False)
            filtered_wv = filtered_wv[name_match | symbol_match]
        
        # Show filtered count
        st.caption(f"📊 Showing **{len(filtered_wv)}** of **{len(wv)}** stocks")
        
        # Edit targets/prices section
        with st.expander("✏️ Edit Buy Prices & Targets | Add Symbol Override", expanded=False):
            st.caption("💡 **Quick Edit:** Adjust buy prices and targets for any stock, or add symbol override for missing prices")
            
            # Two sections side by side
            edit_col, override_col = st.columns(2)
            
            with edit_col:
                st.write("**Edit Buy Prices & Targets**")
                if not wv.empty:
                    stock_options = [f"{row['Name']} ({row['Symbol']})" for _, row in wv.iterrows()]
                    selected_stock_idx = st.selectbox(
                        "Select Stock",
                        options=range(len(stock_options)),
                        format_func=lambda x: stock_options[x],
                        key="edit_stock_select"
                    )
                    
                    if selected_stock_idx is not None:
                        selected_row = wv.iloc[selected_stock_idx]
                        st.write(f"**Current Values for {selected_row['Name']}:**")
                        
                        with st.form("edit_targets_form"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                new_buy_low = st.number_input(
                                    "Buy Low (€)",
                                    value=float(selected_row["Buy Low"]) if pd.notna(selected_row["Buy Low"]) else 0.0,
                                    min_value=0.0,
                                    step=0.01,
                                    format="%.2f",
                                    help="Auto: current price × 0.9"
                                )
                                new_buy_high = st.number_input(
                                    "Buy High (€)",
                                    value=float(selected_row["Buy High"]) if pd.notna(selected_row["Buy High"]) else 0.0,
                                    min_value=0.0,
                                    step=0.01,
                                    format="%.2f",
                                    help="Auto: current price × 1.1"
                                )
                            
                            with col2:
                                new_target_1y = st.number_input(
                                    "Target 1Y (€)",
                                    value=float(selected_row["Target 1Y"]) if pd.notna(selected_row["Target 1Y"]) else 0.0,
                                    min_value=0.0,
                                    step=0.01,
                                    format="%.2f",
                                    help="3Y, 5Y, 10Y auto-calculated from this"
                                )
                            
                            st.caption("ℹ️ Multi-year targets (3Y, 5Y, 10Y) are automatically calculated from Target 1Y using sector growth rates")
                            
                            if st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True):
                                # Update the CSV
                                wl_update = load_csv(WATCHLIST_CSV, WATCHLIST_COLS)
                                mask = (wl_update["Symbol"] == selected_row["Symbol"]) & (wl_update["Name"] == selected_row["Name"])
                                
                                if new_buy_low > 0:
                                    wl_update.loc[mask, "Buy_Low"] = new_buy_low
                                if new_buy_high > 0:
                                    wl_update.loc[mask, "Buy_High"] = new_buy_high
                                if new_target_1y > 0:
                                    wl_update.loc[mask, "Target_1Y"] = new_target_1y
                                
                                save_csv(WATCHLIST_CSV, wl_update)
                                st.success(f"✅ Updated {selected_row['Name']}!")
                                st.rerun()
            
            with override_col:
                st.write("**Add Symbol Override**")
                st.caption("Use this when Current Price is missing - map your symbol to the correct exchange ticker")
                
                with st.form("add_override_form"):
                    # Show stocks with missing prices
                    missing_price_stocks = wv[wv["Current Price"].isna()]
                    if not missing_price_stocks.empty:
                        st.info(f"⚠️ {len(missing_price_stocks)} stocks have missing prices")
                        missing_options = [f"{row['Name']} ({row['Symbol']})" for _, row in missing_price_stocks.iterrows()]
                        selected_missing = st.selectbox(
                            "Stock with missing price",
                            options=missing_options,
                            help="Select a stock that needs symbol override"
                        )
                    else:
                        st.success("✅ All stocks have prices!")
                        selected_missing = None
                    
                    ovr_user_sym = st.text_input(
                        "Your Symbol",
                        value=selected_missing.split("(")[1].replace(")", "").strip() if selected_missing else "",
                        help="The symbol in your watchlist"
                    )
                    ovr_provider_sym = st.text_input(
                        "Provider Symbol",
                        placeholder="e.g., ASML.AS for ASML on Amsterdam",
                        help="The actual symbol used by Marketstack (with exchange suffix)"
                    )
                    ovr_ccy = st.selectbox(
                        "Provider Currency",
                        options=["EUR", "USD", "GBP", "CHF", "CAD", "SEK", "HKD", "PLN"],
                        help="Currency of the provider symbol"
                    )
                    
                    if st.form_submit_button("➕ Add Override", type="primary", use_container_width=True):
                        if ovr_user_sym and ovr_provider_sym:
                            overrides = load_overrides()
                            new_override = pd.DataFrame([{
                                "UserSymbol": ovr_user_sym,
                                "ProviderSymbol": ovr_provider_sym,
                                "ProviderCurrency": ovr_ccy
                            }])
                            overrides = pd.concat([overrides, new_override], ignore_index=True)
                            save_csv(OVERRIDES_CSV, overrides)
                            st.success(f"✅ Added override: {ovr_user_sym} → {ovr_provider_sym}")
                            # Clear price cache to force refresh
                            if 'prices_cache' in st.session_state:
                                st.session_state.prices_cache.clear()
                            st.rerun()
                        else:
                            st.error("Please fill in both symbols")

        # Create display version with all columns
        display_cols = ["Name", "Symbol", "Sector", "Currency", "Current Price", "Buy Low", "Buy High", 
                       "% from Buy Low", "% from Buy High", "Target 1Y", "Upside to 1Y", 
                       "Target 3Y", "Upside to 3Y", "Target 5Y", "Upside to 5Y", 
                       "Target 10Y", "Upside to 10Y"]
        
        display_wv = filtered_wv[display_cols].copy()
        
        st.write("**🔭 Watchlist Stocks** - Click column headers to sort. Use the forms above to edit values.")
        
        st.dataframe(
            display_wv,
            use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Currency": st.column_config.TextColumn("Currency", width="small"),
                "Current Price": st.column_config.NumberColumn(
                    "Current Price (€)",
                    format="€%.2f",
                    width="small"
                ),
                "Buy Low": st.column_config.NumberColumn(
                    "Buy Low (€)",
                    format="€%.2f",
                    width="small"
                ),
                "Buy High": st.column_config.NumberColumn(
                    "Buy High (€)",
                    format="€%.2f",
                    width="small"
                ),
                "% from Buy Low": st.column_config.NumberColumn(
                    "% from Buy Low",
                    format="%.1f%%",
                    width="small"
                ),
                "% from Buy High": st.column_config.NumberColumn(
                    "% from Buy High",
                    format="%.1f%%",
                    width="small"
                ),
                "Target 1Y": st.column_config.NumberColumn(
                    "Target 1Y (€)",
                    format="€%.2f",
                    width="small"
                ),
                "Upside to 1Y": st.column_config.NumberColumn(
                    "Upside to 1Y",
                    format="%.1f%%",
                    width="small"
                ),
                "Target 3Y": st.column_config.NumberColumn(
                    "Target 3Y (€)",
                    format="€%.2f",
                    width="small"
                ),
                "Upside to 3Y": st.column_config.NumberColumn(
                    "Upside to 3Y",
                    format="%.1f%%",
                    width="small"
                ),
                "Target 5Y": st.column_config.NumberColumn(
                    "Target 5Y (€)",
                    format="€%.2f",
                    width="small"
                ),
                "Upside to 5Y": st.column_config.NumberColumn(
                    "Upside to 5Y",
                    format="%.1f%%",
                    width="small"
                ),
                "Target 10Y": st.column_config.NumberColumn(
                    "Target 10Y (€)",
                    format="€%.2f",
                    width="small"
                ),
                "Upside to 10Y": st.column_config.NumberColumn(
                    "Upside to 10Y",
                    format="%.1f%%",
                    width="small"
                ),
            },
            hide_index=True
        )


# ----------------------------
# Analysis
# ----------------------------
with tabs[3]:
    st.subheader("Daily / Weekly / Monthly summaries (EOD cadence)")
    tz = ZoneInfo(CFG.get("app",{}).get("timezone","Europe/Madrid"))
    today = datetime.now(tz).date()
    mfolder = month_folder(today)
    wfolder = guess_week_folder(today)
    day_folder = OUTPUT_DIR / "Analysis" / mfolder / wfolder
    day_folder.mkdir(parents=True, exist_ok=True)

    st.caption("Paste channel URLs or @handles (auto-resolve to channel_id), and/or direct video URLs.")
    ch_str = st.text_area("Channel URLs or channel_ids (one per line)", value="https://www.youtube.com/@Click-Capital\nhttps://www.youtube.com/@StocksToday\nhttps://www.youtube.com/@bravosresearch")
    url_str = st.text_area("Video URLs (optional, one per line)", value="")

    # Resolve channels
    channel_ids = []
    for raw in [x.strip() for x in ch_str.splitlines() if x.strip()]:
        cid = resolve_channel_id(raw) or (raw if raw.startswith("UC") else None)
        if cid: channel_ids.append(cid)

    sources = []
    for cid in channel_ids:
        sources.extend(fetch_latest_from_channel(cid, today))

    urls = [x.strip() for x in url_str.splitlines() if x.strip()]
    for u in urls:
        if is_video_url(u):
            sources.append({"title":"(manual)","url":u,"published":today.isoformat()})

    st.write(f"Found **{len(sources)}** video(s) for analysis.")
    if sources:
        with st.expander("📺 View Found Videos"):
            for i, s in enumerate(sources[:10], 1):  # Show first 10
                st.write(f"{i}. **{s['title']}** - {s['published'][:10]}")
                st.caption(s['url'])
    
    # Summary Browser Section
    st.divider()
    st.subheader("📚 Browse Existing Summaries")
    
    analysis_dir = OUTPUT_DIR / "Analysis"
    if analysis_dir.exists():
        # Get all month folders
        month_folders = [f for f in analysis_dir.iterdir() if f.is_dir()]
        month_folders.sort(reverse=True)  # Most recent first
        
        if month_folders:
            selected_month = st.selectbox("Select Month", [f.name for f in month_folders])
            month_path = analysis_dir / selected_month
            
            # Get week folders in selected month
            week_folders = [f for f in month_path.iterdir() if f.is_dir()]
            week_folders.sort(reverse=True)
            
            if week_folders:
                selected_week = st.selectbox("Select Week", [f.name for f in week_folders])
                week_path = month_path / selected_week
                
                # Show summaries in selected week
                summary_files = list(week_path.glob("*.md"))
                if summary_files:
                    summary_files.sort(key=lambda x: x.name, reverse=True)
                    for summary_file in summary_files:
                        with st.expander(f"📄 {summary_file.name}"):
                            try:
                                content = summary_file.read_text(encoding="utf-8")
                                st.markdown(content)
                                st.download_button(
                                    f"Download {summary_file.name}",
                                    data=content.encode("utf-8"),
                                    file_name=summary_file.name,
                                    mime="text/markdown",
                                    key=f"download_{summary_file.name}"
                                )
                            except Exception as e:
                                st.error(f"Error reading {summary_file.name}: {str(e)}")
                else:
                    st.info("No summaries found in this week folder.")
            else:
                st.info("No week folders found in this month.")
        else:
            st.info("No analysis summaries found yet. Create your first summary below!")
    else:
        st.info("No analysis folder found yet. Create your first summary below!")
    
    st.divider()
    st.subheader("🛠️ Create New Summaries")
    
    if not OPENAI_KEY:
        st.warning("⚠️ OpenAI API key required to create summaries. Please add it in the sidebar.")
        st.info("💡 Summaries are created by clicking the buttons below. They are not generated automatically.")
    else:
        st.info("💡 Click the buttons below to create summaries. Summaries are stored in the output/Analysis folder.")
    
    # Create seed summary from recent channel videos
    if st.button("🌱 Create SEED summary (from last 3 videos per channel)"):
        if not OPENAI_KEY:
            st.error("OpenAI API key required for analysis. Please add it in the sidebar.")
        elif not channel_ids:
            st.error("No channel IDs configured. Please add YouTube channel URLs above.")
        else:
            with st.spinner("Fetching recent videos and transcripts..."):
                # Get last 3 videos from each channel
                all_recent_videos = []
                for cid in channel_ids:
                    # Fetch all recent videos (not just from today)
                    import xml.etree.ElementTree as ET
                    url = channel_rss_url(cid)
                    try:
                        r = requests.get(url, timeout=20)
                        r.raise_for_status()
                        root = ET.fromstring(r.text)
                        ns = {'atom': 'http://www.w3.org/2005/Atom'}
                        channel_videos = []
                        for entry in root.findall('atom:entry', ns):
                            title = entry.find('atom:title', ns).text
                            link = entry.find('atom:link', ns).attrib.get('href')
                            published = entry.find('atom:published', ns).text
                            channel_videos.append({'title': title, 'url': link, 'published': published})
                            if len(channel_videos) >= 3:  # Get only last 3 videos
                                break
                        all_recent_videos.extend(channel_videos)
                    except Exception as e:
                        st.warning(f"Failed to fetch videos from channel {cid}: {str(e)}")
                
                st.write(f"Found {len(all_recent_videos)} recent videos across all channels.")
                
                # Fetch transcripts
                transcripts = []
                successful_videos = []
                for video in all_recent_videos:
                    vid = extract_video_id(video["url"])
                    if not vid:
                        continue
                    tx = fetch_transcript_text(vid)
                    if tx:
                        transcripts.append(tx)
                        successful_videos.append(video)
                        # Save transcript
                        fname = f"Seed_Transcript_{len(transcripts):02d}_{today.strftime('%d.%m.%Y')}.docx"
                        outp = day_folder / fname
                        write_docx(tx, outp)
                
                if not transcripts:
                    st.error("No transcripts could be retrieved from the recent videos.")
                    st.info("This might happen if videos don't have transcripts enabled or are very recent.")
                else:
                    st.success(f"Successfully retrieved {len(transcripts)} transcripts from recent videos!")
                    
                    # Create seed summary
                    summary = summarize_texts(transcripts, OPENAI_KEY, today.strftime('%d.%m.%Y'), mode="seed baseline")
                    md_path = day_folder / f"Seed_Summary_{today.strftime('%d.%m.%Y')}.md"
                    docx_path = day_folder / f"Seed_Summary_{today.strftime('%d.%m.%Y')}.docx"
                    md_path.write_text(summary, encoding="utf-8")
                    write_docx(summary, docx_path)
                    
                    st.success(f"Seed summary created from {len(transcripts)} transcripts!")
                    st.download_button("Download Seed Summary", data=summary.encode("utf-8"),
                                       file_name=md_path.name, mime="text/markdown")
                    
                    with st.expander("📺 Videos Used in Seed Summary"):
                        for i, video in enumerate(successful_videos, 1):
                            st.write(f"{i}. **{video['title']}** - {video['published'][:10]}")
                    
                    with st.expander("Preview Seed Summary"):
                        st.markdown(summary)

    if st.button("📝 Build DAILY summary from transcripts"):
        transcripts = []
        saved = []
        for s in sources:
            vid = extract_video_id(s["url"])
            if not vid: 
                continue
            tx = fetch_transcript_text(vid)
            if tx:
                transcripts.append(tx)
                fname = f"Transcript_{len(saved)+1:02d}_{today.strftime('%d.%m.%Y')}.docx"
                outp = day_folder / fname
                write_docx(tx, outp)
                saved.append(fname)
        if not transcripts:
            st.warning("No transcripts available from the found videos.")
            st.info("This could happen if: 1) Videos don't have transcripts enabled, 2) Videos are very recent and transcripts aren't ready yet, or 3) Channel doesn't provide English transcripts.")
            if sources:
                st.info("💡 Try the DEMO summary above to test the feature, or check back later for transcript availability.")
        else:
            summary = summarize_texts(transcripts, OPENAI_KEY, today.strftime('%d.%m.%Y'), mode="daily")
            md_path = day_folder / f"Daily_Summary_{today.strftime('%d.%m.%Y')}.md"
            docx_path = day_folder / f"Daily_Summary_{today.strftime('%d.%m.%Y')}.docx"
            md_path.write_text(summary, encoding="utf-8")
            write_docx(summary, docx_path)
            st.success(f"Daily summary created in: {md_path.parent}")
            st.download_button("Download Daily .md", data=summary.encode("utf-8"),
                               file_name=md_path.name, mime="text/markdown")

    if st.button("📚 Build WEEKLY summary (roll-up of dailies)"):
        dailies = [p.read_text(encoding="utf-8") for p in day_folder.glob("Daily_Summary_*.md")]
        if not dailies:
            st.warning("No daily summaries in this week folder.")
        else:
            summary = summarize_texts(dailies, OPENAI_KEY, f"Week {wfolder}", mode="weekly")
            md_path = day_folder / f"Weekly_Summary_{today.strftime('%d.%m.%Y')}.md"
            docx_path = day_folder / f"Weekly_Summary_{today.strftime('%d.%m.%Y')}.docx"
            md_path.write_text(summary, encoding="utf-8")
            write_docx(summary, docx_path)
            st.success("Weekly summary created.")

    if st.button("🗓️ Build MONTHLY summary (roll-up of weeks)"):
        month_dir = OUTPUT_DIR / "Analysis" / mfolder
        week_md = []
        for wk in month_dir.glob("*"):
            if wk.is_dir():
                for p in wk.glob("Weekly_Summary_*.md"):
                    week_md.append(p.read_text(encoding="utf-8"))
        if not week_md:
            st.warning("No weekly summaries found in this month folder.")
        else:
            summary = summarize_texts(week_md, OPENAI_KEY, f"Month {mfolder}", mode="monthly")
            md_path = month_dir / f"Monthly_Summary_{today.strftime('%d.%m.%Y')}.md"
            docx_path = month_dir / f"Monthly_Summary_{today.strftime('%d.%m.%Y')}.docx"
            md_path.write_text(summary, encoding="utf-8")
            write_docx(summary, docx_path)
            st.success("Monthly summary created.")

# ----------------------------
# Settings
# ----------------------------
with tabs[4]:
    st.subheader("Industry → Sector Growth (YoY %)")
    st.caption("Edit values. Keep it simple (≤10 industries; ≤6 sectors each).")
    growth = load_growth_table()
    edited = st.data_editor(
        growth,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Industry": st.column_config.TextColumn(help="Top-level category"),
            "Sector": st.column_config.TextColumn(help="Specific segment"),
            "YoY_Growth_%": st.column_config.NumberColumn(format="%.1f", help="Expected yearly growth %"),
        }
    )
    if st.button("💾 Save growth table"):
        save_growth_table(edited)
        st.success("Saved.")

    st.divider()
    st.subheader("Symbol Overrides (manual mapping)")
    st.caption("If a symbol won’t price, map it here to a provider symbol/currency and set a multiplier (e.g., ADR ratio).")
    ov = load_csv(OVERRIDES_CSV, OVERRIDE_COLS)
    ov_edit = st.data_editor(ov, use_container_width=True, num_rows="dynamic")
    if st.button("💾 Save overrides"):
        save_csv(OVERRIDES_CSV, ov_edit)
        st.success("Overrides saved.")

    st.divider()
    st.subheader("Preferences")
    st.write("- **Display Currency**: fixed to EUR (auto FX via Marketstack; ECB fallback)")
    st.write("- **Theme**: dark (set in Streamlit if desired)")
