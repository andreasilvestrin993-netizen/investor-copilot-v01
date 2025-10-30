import os, io, math, json, time, re, datetime as dt
from datetime import datetime, date
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st
import pandas as pd
import requests
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

# Session state for performance caching
if 'symbol_cache' not in st.session_state:
    st.session_state.symbol_cache = {}
if 'prices_cache' not in st.session_state:
    st.session_state.prices_cache = {}
if 'last_price_fetch' not in st.session_state:
    st.session_state.last_price_fetch = None

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

PORTFOLIO_COLS = ["Name","Symbol","ISIN","Industry","Sector","Country","Currency","Current_Price","Quantity","BEP","Value_EUR","PL_EUR","PL_Percent"]
WATCHLIST_COLS = ["Name","Symbol","ISIN","Industry","Sector","Country","Currency","Current_Price","Buy_Low","Buy_High","Target_1Y","Target_3Y","Target_5Y","Target_10Y","Delta_Buy_Low","Delta_1Y"]
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
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=cols)
        return ensure_columns(df, cols)
    return pd.DataFrame(columns=cols)

def save_csv(path, df):
    df.to_csv(path, index=False)

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
    Expand easily if you need more.
    Fallback to ECB if missing.
    """
    fx = {"EUR": 1.0}
    if not marketstack_key:
        return fx

    def latest_fx(pair):
        base_url = "http://api.marketstack.com/v1/eod/latest"
        data = ms_get(base_url, {"access_key": marketstack_key, "symbols": pair})
        items = data.get("data", [])
        if items:
            close = items[0].get("close")
            return float(close) if close is not None else None
        return None

    # EURUSD = USD per 1 EUR ; USD->EUR = 1 / EURUSD
    eurusd = latest_fx("EURUSD")
    if eurusd and eurusd != 0:
        fx["USD"] = 1.0 / eurusd

    eurgbp = latest_fx("EURGBP")
    if eurgbp and eurgbp != 0:
        fx["GBP"] = 1.0 / eurgbp

    eurchf = latest_fx("EURCHF")
    if eurchf and eurchf != 0:
        fx["CHF"] = 1.0 / eurchf

    # Fallback to ECB if some missing
    if CFG.get("pricing",{}).get("ecb_fallback", True):
        missing = [k for k in ["USD","GBP","CHF"] if k not in fx]
        if missing:
            try:
                # ECB Daily (JSON via Frankfurter)
                r = requests.get("https://api.frankfurter.app/latest?from=EUR", timeout=15)
                r.raise_for_status()
                rates = r.json().get("rates", {})
                for c in missing:
                    if c in rates and rates[c] != 0:
                        # rates[c] = c per 1 EUR ; want 1 ccy -> EUR => 1 / rates[c]
                        fx[c] = 1.0 / float(rates[c])
            except Exception:
                pass

    return fx

def fetch_eod_prices(symbols, marketstack_key: str) -> dict:
    """Return dict {ProviderSymbol: close} using Marketstack EOD latest with caching."""
    if not symbols or not marketstack_key: 
        return {}
    
    # Check cache first (5-minute expiry)
    now = datetime.now()
    if (st.session_state.last_price_fetch and 
        (now - st.session_state.last_price_fetch).total_seconds() < 300):
        # Return cached prices for requested symbols
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
        for item in data.get("data", []):
            sym = item.get("symbol")
            close = item.get("close")
            if sym and close is not None:
                price = float(close)
                out[sym] = price
                # Update cache
                st.session_state.prices_cache[sym] = price
        time.sleep(0.2)
    
    # Update cache timestamp
    st.session_state.last_price_fetch = now
    return out

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
        symbol_row = sorted(symbol_row, key=lambda x: 0 if (x.get("name","").lower().find(name_hint_low) >= 0) else 1)
    # prefer has_eod
    symbol_row = sorted(symbol_row, key=lambda x: 0 if x.get("has_eod") else 1)
    return symbol_row[0]

def load_overrides():
    return load_csv(OVERRIDES_CSV, OVERRIDE_COLS)

def resolve_provider_symbol(user_symbol: str, name_hint: str, ccy: str, marketstack_key: str, prices_cache: dict) -> tuple[str,str]:
    """
    Return (provider_symbol, provider_ccy)
    Priority:
      1) Manual override in symbol_overrides.csv
      2) Direct user_symbol if it prices
      3) Search by symbol; else search by name hint; pick best; use its 'symbol'
    """
    overrides = load_overrides()
    ov = overrides[overrides["UserSymbol"].fillna("").str.upper() == (user_symbol or "").upper()]
    if not ov.empty:
        row = ov.iloc[0]
        psym = (row.get("ProviderSymbol") or "").strip()
        pccy = (row.get("ProviderCurrency") or ccy or "EUR").strip() or "EUR"
        return psym or user_symbol, pccy

    # If the direct symbol already priced in cache, use it
    if user_symbol in prices_cache:
        return user_symbol, ccy or "EUR"

    # Search by symbol first
    cand = search_alternative_symbols(user_symbol, marketstack_key, limit=5)
    best = pick_best_alt(cand, name_hint)
    if best and best.get("symbol"):
        return best["symbol"], (best.get("currency") or ccy or "EUR")

    # Try by name hint if present
    if name_hint:
        cand2 = search_alternative_symbols(name_hint, marketstack_key, limit=5)
        best2 = pick_best_alt(cand2, name_hint)
        if best2 and best2.get("symbol"):
            return best2["symbol"], (best2.get("currency") or ccy or "EUR")

    # Give up: return original symbol
    return user_symbol, ccy or "EUR"

def to_eur(amount, ccy, fx_map):
    if amount is None or pd.isna(amount): return None
    if (not ccy) or ccy.upper() == "EUR": return float(amount)
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
    FX_MAP = fetch_fx_map_eur(MARKETSTACK_KEY) if MARKETSTACK_KEY else {"EUR": 1.0}
    st.sidebar.caption(f"FX loaded (→ EUR): {', '.join([f'{k}' for k in FX_MAP.keys()])}")
except Exception as e:
    FX_MAP = {"EUR": 1.0}
    st.sidebar.error(f"FX fetch failed: {str(e)}")

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
        # Build provider symbols with fallback/overrides - optimized batch approach
        symbols_direct = [str(s) for s in port["Symbol"].dropna().unique().tolist()]
        prices_direct = fetch_eod_prices(symbols_direct, MARKETSTACK_KEY) if MARKETSTACK_KEY else {}

        # Collect all provider symbols needed for analysis
        all_analysis_symbols = set(symbols_direct)
        for _, r in port.iterrows():
            user_sym = str(r["Symbol"]) if pd.notna(r["Symbol"]) else ""
            name_hint = str(r["Name"] or "") if pd.notna(r["Name"]) else ""
            user_ccy = str(r["Currency"] or "EUR").upper() if pd.notna(r["Currency"]) else "EUR"
            if user_sym and user_sym not in prices_direct:
                psym, _ = resolve_provider_symbol(user_sym, name_hint, user_ccy, MARKETSTACK_KEY, prices_direct)
                if psym:
                    all_analysis_symbols.add(psym)
        
        # Fetch all missing prices in one batch
        missing_analysis_symbols = [sym for sym in all_analysis_symbols if sym not in prices_direct]
        if missing_analysis_symbols and MARKETSTACK_KEY:
            additional_analysis_prices = fetch_eod_prices(missing_analysis_symbols, MARKETSTACK_KEY)
            prices_direct.update(additional_analysis_prices)

        total_val_eur = 0.0
        total_cost_eur = 0.0

        for _, r in port.iterrows():
            qty = float(r["Quantity"] or 0)
            bep = float(r["BEP"] or 0.0)
            user_ccy = str(r["Currency"] or "EUR").upper() if pd.notna(r["Currency"]) else "EUR"
            user_sym = str(r["Symbol"]) if pd.notna(r["Symbol"]) else ""
            name_hint = str(r["Name"] or "") if pd.notna(r["Name"]) else ""

            # Resolve provider symbol if not priced
            if user_sym not in prices_direct:
                psym, pccy = resolve_provider_symbol(user_sym, name_hint, user_ccy, MARKETSTACK_KEY, prices_direct)
            else:
                psym, pccy = user_sym, user_ccy

            # Read price and convert
            last = prices_direct.get(psym, None)
            last_eur = to_eur(last, pccy, FX_MAP) if last is not None else None

            # Convert BEP (user native) to EUR
            bep_eur = to_eur(bep, user_ccy, FX_MAP) if bep else None

            if last_eur is not None and bep_eur is not None:
                total_val_eur += qty * last_eur
                total_cost_eur += qty * bep_eur

        total_pnl_eur = total_val_eur - total_cost_eur
        pnl_pct = (total_pnl_eur / total_cost_eur * 100) if total_cost_eur > 0 else 0.0

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Portfolio Value", format_large_number(total_val_eur))
        c2.metric("Total P/L", format_large_number(total_pnl_eur), format_percentage(pnl_pct))
        c3.metric("Positions", f"{len(port)}")
        c4.metric("Display Currency", BASE_CCY)

# ----------------------------
# Portfolio
# ----------------------------
with tabs[1]:
    st.subheader("Portfolio (EUR view, EOD, auto FX, alt listing fallback)")
    st.caption("Columns: Name, Symbol, Currency, Quantity, BEP, Sector, Industry, Multiplier (optional; e.g., ADR ratio)")

    templ = pd.DataFrame(columns=PORTFOLIO_COLS)
    st.download_button("Download CSV template", templ.to_csv(index=False).encode("utf-8"),
                       "portfolio_template.csv", "text/csv")

    up = st.file_uploader("Import CSV", type=["csv"], key="port_upl")
    if up:
        try:
            # Read CSV with better handling for decimal numbers and mixed types
            df = pd.read_csv(up, dtype={
                'Name': 'str',
                'Symbol': 'str', 
                'ISIN': 'str',
                'Currency': 'str',
                'Quantity': 'float64',
                'BEP': 'float64',
                'Value': 'float64',
                'Total P/L€': 'float64',
                'Total P/L%': 'float64'
            }, na_values=['', 'None', 'NaN', 'null'])
            
            # Ensure we have the right columns
            df = ensure_columns(df, PORTFOLIO_COLS)
            
            # Clean up any potential parsing issues
            for col in ['Quantity', 'BEP', 'Value', 'Total P/L€', 'Total P/L%']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            save_csv(PORTFOLIO_CSV, df)
            st.success(f"Portfolio imported successfully! {len(df)} rows loaded.")
            
            # Show preview of imported data
            with st.expander("📋 Preview Imported Data"):
                st.dataframe(df.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error importing CSV: {str(e)}")
            st.info("Please ensure your CSV has the correct column structure and decimal numbers use '.' not ','.")

    port = load_csv(PORTFOLIO_CSV, PORTFOLIO_COLS)

    with st.expander("Add / Edit Row"):
        with st.form("add_port"):
            st.write("**Basic Information**")
            c1,c2,c3 = st.columns(3)
            name = c1.text_input("Name*", help="Company name")
            symbol = c2.text_input("Symbol*", help="Stock ticker (e.g., NVDA)")
            isin = c3.text_input("ISIN", help="Optional: International Securities ID")
            
            st.write("**Classification**")
            c4,c5,c6 = st.columns(3)
            industry = c4.text_input("Industry*", help="e.g., Software, Semiconductors")
            sector = c5.text_input("Sector*", help="e.g., Technology, Healthcare")
            country = c6.text_input("Country*", help="e.g., USA, Germany")
            
            st.write("**Financial Details**")
            c7,c8 = st.columns(2)
            currency = c7.selectbox("Currency*", ["EUR","USD","GBP","CHF"])
            current_price = c8.number_input("Current Price", min_value=0.0, step=0.01, format="%.2f", help="Will be auto-updated from API")
            
            c9,c10 = st.columns(2)
            qty = c9.number_input("Quantity*", min_value=0.0, step=1.0)
            bep = c10.number_input("BEP*", min_value=0.0, step=0.01, format="%.2f", help="Break-even price per share")
            
            st.caption("Fields marked with * are required. Value, P/L amounts and percentages are calculated automatically.")
            add_btn = st.form_submit_button("Save Position")
        if add_btn:
            if not name or not symbol or not industry or not sector or not country or not qty or not bep:
                st.error("Please fill in all required fields marked with *")
            else:
                new = pd.DataFrame([{
                    "Name": name,
                    "Symbol": symbol,
                    "ISIN": isin,
                    "Industry": industry,
                    "Sector": sector,
                    "Country": country,
                    "Currency": currency,
                    "Current_Price": current_price if current_price > 0 else None,
                    "Quantity": qty,
                    "BEP": bep,
                    "Value_EUR": None,  # Will be calculated
                    "PL_EUR": None,     # Will be calculated  
                    "PL_Percent": None  # Will be calculated
                }])
                port = pd.concat([port, new], ignore_index=True)
                save_csv(PORTFOLIO_CSV, port)
            st.success("Saved.")

    # Action buttons
    col_refresh, col_clear, col_info = st.columns([1, 1, 3])
    with col_refresh:
        if st.button("🔄 Refresh Prices", help="Force refresh current prices (cache expires after 5 minutes)"):
            st.session_state.prices_cache.clear()
            st.session_state.last_price_fetch = None
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

    # Pricing block with fallback - first collect all provider symbols
    with st.spinner("Loading current prices..."):
        syms = [str(s) for s in port["Symbol"].dropna().unique().tolist()]
        prices = fetch_eod_prices(syms, MARKETSTACK_KEY) if MARKETSTACK_KEY and syms else {}
    
    # Collect all possible provider symbols needed
    all_provider_symbols = set()
    for _, row in port.iterrows():
        user_sym = str(row["Symbol"]) if pd.notna(row["Symbol"]) else ""
        name_hint = str(row["Name"] or "") if pd.notna(row["Name"]) else ""
        user_ccy = str(row["Currency"] or "EUR").upper() if pd.notna(row["Currency"]) else "EUR"
        if user_sym:
            psym, _ = resolve_provider_symbol(user_sym, name_hint, user_ccy, MARKETSTACK_KEY, prices)
            if psym:
                all_provider_symbols.add(psym)
    
    # Fetch prices for missing provider symbols in one batch
    missing_symbols = [sym for sym in all_provider_symbols if sym not in prices]
    if missing_symbols and MARKETSTACK_KEY:
        additional_prices = fetch_eod_prices(missing_symbols, MARKETSTACK_KEY)
        prices.update(additional_prices)

    def calc_row(_r):
        qty = float(_r["Quantity"] or 0)
        user_sym = str(_r["Symbol"]) if pd.notna(_r["Symbol"]) else ""
        name_hint = str(_r["Name"] or "") if pd.notna(_r["Name"]) else ""
        user_ccy = str(_r["Currency"] or "EUR").upper() if pd.notna(_r["Currency"]) else "EUR"
        bep = float(_r["BEP"] or 0.0)
        psym, pccy = resolve_provider_symbol(user_sym, name_hint, user_ccy, MARKETSTACK_KEY, prices)

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

        # Create a clean display dataframe with proper column order
        display_df = pd.DataFrame({
            "Name": view["Name"],
            "Symbol": view["Symbol"], 
            "ISIN": view["ISIN"],
            "Currency": view["Currency"],
            "Quantity": view["Quantity"],
            "BEP": view["BEP"],
            "Current Price": [format_price(x) if x is not None else "-" for x in last_eurs],
            "Value": [format_large_number(x) if x is not None else "-" for x in vals],
            "Total P/L€": [format_large_number(x) if x is not None else "-" for x in pnls],
            "Total P/L%": [format_percentage(x) if x is not None else "-" for x in pnl_pcts]
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
    
    # Display the clean, well-formatted portfolio
    st.dataframe(display_df, use_container_width=True)
    
    # Add diagnostic columns for debugging (can be hidden later)
    if not port.empty and MARKETSTACK_KEY:
        with st.expander("🔧 Advanced Info (Provider Data)"):
            debug_view = port.copy()
            if provs:  # Only add if we have data
                debug_view["ProviderSymbol"] = provs
                debug_view["ProviderCcy"] = pccys
                debug_view["EffMultiplier"] = mults
                debug_view["Last (EUR)"] = [fmt_price(x) if x is not None else "-" for x in last_eurs]
            st.dataframe(debug_view, use_container_width=True)
    
    # Button to save updated portfolio with calculated values
    if not port.empty and MARKETSTACK_KEY:
        if st.button("💾 Save updated portfolio (with calculated values)"):
            save_csv(PORTFOLIO_CSV, view)  # Save the original structure with updated values
            st.success("Portfolio saved with updated values!")

    st.caption("If a symbol is mispriced or missing, add a row in **symbol_overrides.csv** (UserSymbol → ProviderSymbol, ProviderCurrency, Multiplier).")

# ----------------------------
# Watchlists
# ----------------------------
with tabs[2]:
    st.subheader("Watchlists (targets & distance, EUR)")
    st.caption("Columns: List, Name, Symbol, Currency, BuyLow, BuyHigh, Sector, Industry, Target1Y, Target3Y, Target5Y, Target10Y, Multiplier")

    templw = pd.DataFrame(columns=WATCHLIST_COLS)
    st.download_button("Download CSV template", templw.to_csv(index=False).encode("utf-8"),
                       "watchlist_template.csv", "text/csv")

    upw = st.file_uploader("Import CSV", type=["csv"], key="wl_upl")
    if upw:
        try:
            # Read CSV with proper data types for watchlist
            dfw = pd.read_csv(upw, dtype={
                'Name': 'str',
                'Symbol': 'str',
                'Currency': 'str',
                'BuyLow': 'float64',
                'BuyHigh': 'float64',
                'Sector': 'str',
                'Industry': 'str',
                'Target1Y': 'float64',
                'Target3Y': 'float64',
                'Target5Y': 'float64',
                'Target10Y': 'float64'
            }, na_values=['', 'None', 'NaN', 'null'])
            
            dfw = ensure_columns(dfw, WATCHLIST_COLS)
            
            # Clean up numeric columns
            for col in ['BuyLow', 'BuyHigh', 'Target1Y', 'Target3Y', 'Target5Y', 'Target10Y']:
                if col in dfw.columns:
                    dfw[col] = pd.to_numeric(dfw[col], errors='coerce').fillna(0.0)
            
            save_csv(WATCHLIST_CSV, dfw)
            st.success(f"Watchlist imported successfully! {len(dfw)} rows loaded.")
            
        except Exception as e:
            st.error(f"Error importing watchlist CSV: {str(e)}")
            st.info("Please ensure your CSV has the correct column structure and decimal numbers use '.' not ','.")

    wl = load_csv(WATCHLIST_CSV, WATCHLIST_COLS)

    with st.expander("Add / Edit Row"):
        with st.form("add_wl"):
            st.write("**Basic Information**")
            c1,c2,c3 = st.columns(3)
            wname = c1.text_input("Name*", help="Company name")
            wsym = c2.text_input("Symbol*", help="Stock ticker (e.g., NVDA)")
            wisin = c3.text_input("ISIN", help="Optional: International Securities ID")
            
            st.write("**Classification**")
            c4,c5,c6 = st.columns(3)
            wind = c4.text_input("Industry*", help="e.g., Software, Semiconductors")
            wsec = c5.text_input("Sector*", help="e.g., Technology, Healthcare")
            wcountry = c6.text_input("Country*", help="e.g., USA, Germany")
            
            st.write("**Price Information**")
            c7,c8 = st.columns(2)
            wccy = c7.selectbox("Currency*", ["EUR","USD","GBP","CHF"])
            wcurrent = c8.number_input("Current Price", min_value=0.0, step=0.01, format="%.2f", help="Will be auto-updated from API")
            
            st.write("**Buy Targets**")
            d1,d2 = st.columns(2)
            buyl = d1.number_input("Buy Low*", min_value=0.0, step=0.01, format="%.2f", help="Lower buy target")
            buyh = d2.number_input("Buy High*", min_value=0.0, step=0.01, format="%.2f", help="Higher buy target")
            
            st.write("**Price Targets**")
            t1,t2,t3,t4 = st.columns(4)
            t1y = t1.number_input("Target 1Y", min_value=0.0, step=0.01, format="%.2f")
            t3y = t2.number_input("Target 3Y", min_value=0.0, step=0.01, format="%.2f")
            t5y = t3.number_input("Target 5Y", min_value=0.0, step=0.01, format="%.2f")
            t10y = t4.number_input("Target 10Y", min_value=0.0, step=0.01, format="%.2f")
            
            st.caption("Fields marked with * are required. Delta values are calculated automatically.")
            addw = st.form_submit_button("Save to Watchlist")
        if addw:
            if not wname or not wsym or not wind or not wsec or not wcountry or not buyl or not buyh:
                st.error("Please fill in all required fields marked with *")
            else:
                # Calculate delta values
                delta_buy_low = None  # Will be calculated with current price
                delta_1y = None      # Will be calculated with current price
                
                neww = pd.DataFrame([{
                    "Name": wname,
                    "Symbol": wsym,
                    "ISIN": wisin,
                    "Industry": wind,
                    "Sector": wsec,
                    "Country": wcountry,
                    "Currency": wccy,
                    "Current_Price": wcurrent if wcurrent > 0 else None,
                    "Buy_Low": buyl,
                    "Buy_High": buyh,
                    "Target_1Y": t1y if t1y > 0 else None,
                    "Target_3Y": t3y if t3y > 0 else None,
                    "Target_5Y": t5y if t5y > 0 else None,
                    "Target_10Y": t10y if t10y > 0 else None,
                    "Delta_Buy_Low": delta_buy_low,
                    "Delta_1Y": delta_1y
                }])
                wl = pd.concat([wl, neww], ignore_index=True)
                save_csv(WATCHLIST_CSV, wl)
                st.success("Added to watchlist!")

    # Price refresh with alt listing fallback - collect all provider symbols first
    syms_w = [str(s) for s in wl["Symbol"].dropna().unique().tolist()]
    prices_w = fetch_eod_prices(syms_w, MARKETSTACK_KEY) if MARKETSTACK_KEY and syms_w else {}
    
    # Collect all possible provider symbols needed for watchlist
    all_provider_symbols_w = set()
    for _, row in wl.iterrows():
        user_sym = str(row["Symbol"]) if pd.notna(row["Symbol"]) else ""
        name_hint = str(row["Name"] or "") if pd.notna(row["Name"]) else ""
        ccy = str(row["Currency"] or "EUR").upper() if pd.notna(row["Currency"]) else "EUR"
        if user_sym:
            psym, _ = resolve_provider_symbol(user_sym, name_hint, ccy, MARKETSTACK_KEY, prices_w)
            if psym:
                all_provider_symbols_w.add(psym)
    
    # Fetch prices for missing provider symbols in one batch
    missing_symbols_w = [sym for sym in all_provider_symbols_w if sym not in prices_w]
    if missing_symbols_w and MARKETSTACK_KEY:
        additional_prices_w = fetch_eod_prices(missing_symbols_w, MARKETSTACK_KEY)
        prices_w.update(additional_prices_w)

    def wl_calc_row(r):
        user_sym = str(r["Symbol"]) if pd.notna(r["Symbol"]) else ""
        name_hint = str(r["Name"] or "") if pd.notna(r["Name"]) else ""
        ccy = str(r["Currency"] or "EUR").upper() if pd.notna(r["Currency"]) else "EUR"
        psym, pccy = resolve_provider_symbol(user_sym, name_hint, ccy, MARKETSTACK_KEY, prices_w)

        # Now we should have all prices already - no individual API calls needed
        last = prices_w.get(psym, None)
        last_eur = to_eur(last, pccy, FX_MAP) if last is not None else None

        def fnum(x):
            return float(x) if x is not None and not pd.isna(x) else None

        buyl = fnum(r.get("Buy_Low"))
        buyh = fnum(r.get("Buy_High"))
        t1  = fnum(r.get("Target_1Y"))
        t3  = fnum(r.get("Target_3Y"))
        t5  = fnum(r.get("Target_5Y"))
        t10 = fnum(r.get("Target_10Y"))

        from_low = ((last_eur - buyl) / buyl * 100) if (last_eur is not None and buyl and buyl>0) else None
        from_high = ((last_eur - buyh) / buyh * 100) if (last_eur is not None and buyh and buyh>0) else None
        to_1y = ((t1 - last_eur) / last_eur * 100) if (t1 and last_eur and last_eur>0) else None
        to_3y = ((t3 - last_eur) / last_eur * 100) if (t3 and last_eur and last_eur>0) else None
        to_5y = ((t5 - last_eur) / last_eur * 100) if (t5 and last_eur and last_eur>0) else None
        to_10y = ((t10 - last_eur) / last_eur * 100) if (t10 and last_eur and last_eur>0) else None

        return last_eur, from_low, from_high, to_1y, to_3y, to_5y, to_10y, psym, pccy

    wv = wl.copy()
    last_eur_list, d_low, d_high, d1, d3, d5, d10, ps, pc = [], [], [], [], [], [], [], [], []
    for _, r in wv.iterrows():
        le, fl, fh, t1, t3, t5, t10, pps, ppc = wl_calc_row(r)
        last_eur_list.append(le); d_low.append(fl); d_high.append(fh); d1.append(t1); d3.append(t3); d5.append(t5); d10.append(t10)
        ps.append(pps); pc.append(ppc)

    # Update the existing Current Price column and add calculated columns
    wv["Current Price"] = [format_price(x) if x is not None else "-" for x in last_eur_list]
    wv["% from BuyLow"] = [format_percentage(x) if x is not None else "-" for x in d_low]
    wv["% from BuyHigh"] = [format_percentage(x) if x is not None else "-" for x in d_high]
    wv["Upside to 1Y"] = [format_percentage(x) if x is not None else "-" for x in d1]
    wv["Upside to 3Y"] = [format_percentage(x) if x is not None else "-" for x in d3]
    wv["Upside to 5Y"] = [format_percentage(x) if x is not None else "-" for x in d5]
    wv["Upside to 10Y"] = [format_percentage(x) if x is not None else "-" for x in d10]

    # Add debug info for watchlist too
    with st.expander("🔧 Watchlist Advanced Info"):
        debug_wv = wv.copy()
        debug_wv["ProviderSymbol"] = ps
        debug_wv["ProviderCcy"] = pc
        st.dataframe(debug_wv, use_container_width=True)

    st.dataframe(wv, use_container_width=True)
    st.caption("If a symbol is mispriced/missing, add to **symbol_overrides.csv** and click refresh.")

    if st.button("🔄 Refresh EOD prices"):
        st.success("EOD prices updated (EUR view).")

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
