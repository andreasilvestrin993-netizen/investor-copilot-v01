import os, io, math, json, time, re, datetime as dt
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from docx import Document
from dotenv import load_dotenv

# ----------------------------
# Modular imports
# ----------------------------
from config.settings import (
    APP_DIR, DATA_DIR, OUTPUT_DIR, CACHE_DIR, BASE_CCY, CFG,
    PRICES_CACHE_FILE, FX_CACHE_FILE, PRICES_CACHE_TTL, FX_CACHE_TTL,
    IND_GROWTH_CSV, PORTFOLIO_CSV, WATCHLIST_CSV, OVERRIDES_CSV,
    PORTFOLIO_COLS, WATCHLIST_COLS, OVERRIDE_COLS
)
from utils.cache import load_daily_cache, save_daily_cache
from utils.calculations import calc_eagr_targets, format_price, format_percent, format_quantity
from utils.formatters import format_percentage, format_large_number, clamp_format_number, fmt_pct
from utils.csv_utils import (
    detect_csv_format, read_csv_smart, show_column_mapping_ui,
    ensure_columns, load_csv, save_csv, write_docx
)
from utils.sector_utils import (
    get_sector_list, map_industry_to_sector, load_growth_table, save_growth_table,
    init_industry_growth_csv
)
from utils.helpers import get_secret, fmt_price, guess_week_folder, month_folder
from services.marketstack import (
    fetch_fx_map_eur, fetch_eod_prices, fetch_52week_data,
    resolve_provider_symbol, search_companies, to_eur
)
from services.youtube_service import (
    is_video_url, extract_video_id, resolve_channel_id,
    fetch_latest_from_channel, fetch_transcript_text
)
from services.openai_service import summarize_texts

# ----------------------------
# Bootstrap
# ----------------------------
load_dotenv()

# Initialize industry growth CSV if needed
init_industry_growth_csv()

# Initialize overrides CSV if needed
if not OVERRIDES_CSV.exists():
    pd.DataFrame(columns=OVERRIDE_COLS).to_csv(OVERRIDES_CSV, index=False)

# Session state for performance caching
if 'symbol_cache' not in st.session_state:
    st.session_state.symbol_cache = {}
if 'prices_cache' not in st.session_state:
    st.session_state.prices_cache = load_daily_cache(PRICES_CACHE_FILE, PRICES_CACHE_TTL)
if 'last_price_fetch' not in st.session_state:
    # Set to today if we loaded cache, otherwise None
    if st.session_state.prices_cache:
        st.session_state.last_price_fetch = datetime.now()
    else:
        st.session_state.last_price_fetch = None
if 'fx_cache' not in st.session_state:
    st.session_state.fx_cache = load_daily_cache(FX_CACHE_FILE, FX_CACHE_TTL)
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
# All utility functions moved to modular imports above
# ----------------------------

from services.marketstack import fetch_eod_prices as _ms_fetch_eod_prices

def fetch_eod_prices(symbols, marketstack_key: str) -> dict:
    return _ms_fetch_eod_prices(symbols, marketstack_key)

from services.marketstack import fetch_52week_data as _ms_fetch_52w

def fetch_52week_data(symbols, marketstack_key: str) -> dict:
    return _ms_fetch_52w(symbols, marketstack_key)

from services.marketstack import search_alternative_symbols as _ms_search_alts

def search_alternative_symbols(query: str, marketstack_key: str, limit=5):
    return _ms_search_alts(query, marketstack_key, limit)

from services.marketstack import pick_best_alt as _ms_pick_best_alt

def pick_best_alt(symbol_row, name_hint=None):
    return _ms_pick_best_alt(symbol_row, name_hint)

from services.marketstack import load_overrides as _ms_load_overrides

def load_overrides():
    return _ms_load_overrides()

from services.marketstack import save_override as _ms_save_override

def save_override(user_symbol: str, provider_symbol: str, provider_ccy: str):
    return _ms_save_override(user_symbol, provider_symbol, provider_ccy)

from services.marketstack import resolve_provider_symbol as _ms_resolve

def resolve_provider_symbol(user_symbol: str, name_hint: str, ccy: str, marketstack_key: str, prices_cache: dict, auto_save: bool = False) -> tuple[str,str]:
    return _ms_resolve(user_symbol, name_hint, ccy, marketstack_key, prices_cache, auto_save)

from services.marketstack import search_companies as _ms_search_companies

def search_companies(query: str, marketstack_key: str, limit=10):
    return _ms_search_companies(query, marketstack_key, limit)

from services.marketstack import to_eur as _ms_to_eur

def to_eur(amount, ccy, fx_map):
    return _ms_to_eur(amount, ccy, fx_map)

# ----------------------------
# YouTube helpers (RSS + transcript; no Google API)
# ----------------------------
from services.youtube_service import is_video_url as _yt_is_video_url

def is_video_url(url: str) -> bool:
    return _yt_is_video_url(url)

from services.youtube_service import extract_video_id as _yt_extract_video_id

def extract_video_id(url: str) -> str | None:
    return _yt_extract_video_id(url)

def channel_rss_url(channel_id: str) -> str:
    return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

from services.youtube_service import resolve_channel_id as _yt_resolve_channel_id

def resolve_channel_id(channel_url_or_handle: str) -> str | None:
    return _yt_resolve_channel_id(channel_url_or_handle)

from services.youtube_service import fetch_latest_from_channel as _yt_fetch_latest

def fetch_latest_from_channel(channel_id: str, since_date: date) -> list[dict]:
    return _yt_fetch_latest(channel_id, since_date)

from services.youtube_service import fetch_transcript_text as _yt_fetch_transcript

def fetch_transcript_text(video_id: str) -> str | None:
    return _yt_fetch_transcript(video_id)

# ----------------------------
# OpenAI summarization (delegated to service)
# ----------------------------
from services.openai_service import summarize_texts as _summarize_texts_service

def summarize_texts(texts: list[str], openai_key: str, date_str: str, mode="daily"):
    return _summarize_texts_service(texts, openai_key, date_str, mode)

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
            
            # Get 52-week data for this symbol if available
            week52_info = week52_data.get(psym, {})
            week52_high = week52_info.get('high')
            week52_low = week52_info.get('low')
            
            # Calculate 52-week momentum (price change %)
            r52w_pct = None
            if week52_low and last and week52_low > 0:
                r52w_pct = ((last - week52_low) / week52_low) * 100

            def fnum(x):
                return float(x) if x is not None and not pd.isna(x) else None

            # Get sector growth rate
            growth_rate_pct = None
            if sector:
                growth_df = load_csv(IND_GROWTH_CSV, ["Industry","Sector","YoY_Growth_%"])
                sector_rows = growth_df[growth_df["Sector"].str.upper() == sector.upper()]
                if not sector_rows.empty:
                    growth_rate_pct = sector_rows["YoY_Growth_%"].mean()  # Keep as percentage

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
            
            # Auto-populate Target 1Y if not set (using simple sector growth)
            if t1 is None and last_eur is not None and growth_rate_pct is not None:
                t1 = last_eur * (1 + (growth_rate_pct / 100))
            
            # Calculate multi-year targets using EAGR formula
            if t1 and last_eur and growth_rate_pct is not None:
                t1_adj, t3_calc, t5_calc, t10_calc = calc_eagr_targets(
                    current_price=last_eur,
                    target_1y=t1,
                    sector_growth_pct=growth_rate_pct,
                    r52w_pct=r52w_pct
                )
                
                # Use EAGR calculations for multi-year targets
                if t3 is None and t3_calc:
                    t3 = t3_calc
                if t5 is None and t5_calc:
                    t5 = t5_calc
                if t10 is None and t10_calc:
                    t10 = t10_calc

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
                            
                            st.caption("ℹ️ Multi-year targets (3Y, 5Y, 10Y) are automatically calculated using EAGR formula: 70% sector growth + 30% 52W momentum, bounded -10% to +25%")
                            
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
        
        # Apply color coding based on opportunity zones
        def highlight_opportunities(row):
            """
            Color code rows based on buy zone proximity and upside potential:
            - Green: Near buy zone (price within 5% of Buy High)
            - Gold: Moderate upside (10-20% to target)
            - Red: Overvalued (price above 1Y target)
            - White/Default: Normal (>20% to target)
            """
            current = row.get("Current Price")
            buy_high = row.get("Buy High")
            target_1y = row.get("Target 1Y")
            upside_1y = row.get("Upside to 1Y")
            
            # Skip if key values missing
            if pd.isna(current) or pd.isna(buy_high) or pd.isna(target_1y):
                return [''] * len(row)
            
            # Determine color based on opportunity zone
            if current <= buy_high * 1.05:  # Within 5% of buy zone
                color = 'background-color: rgba(76, 175, 80, 0.15)'  # Green
            elif current >= target_1y:  # Overvalued
                color = 'background-color: rgba(244, 67, 54, 0.15)'  # Red
            elif upside_1y and upside_1y < 20:  # Moderate upside
                color = 'background-color: rgba(255, 193, 7, 0.15)'  # Gold/Yellow
            else:
                color = ''  # Default (white/dark depending on theme)
            
            return [color] * len(row)
        
        st.write("**🔭 Watchlist Stocks** - Click column headers to sort. Color-coded by opportunity.")
        st.caption("🟢 Green: Near buy zone | 🟡 Gold: Moderate upside | 🔴 Red: Above 1Y target | ⚪ White: Normal")
        
        # Apply styling with proper error handling
        try:
            styled_wv = display_wv.style.apply(highlight_opportunities, axis=1)
            st.dataframe(
                styled_wv,
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
        except Exception as e:
            # Fallback to plain dataframe if styling fails
            st.warning("⚠️ Color-coding not available, showing plain table")
            st.dataframe(
                display_wv,
                use_container_width=True,
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
    st.caption("Edit values. Rules: All fields required, growth must be between -50% and +100%.")
    growth = load_growth_table()
    edited = st.data_editor(
        growth,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Industry": st.column_config.TextColumn(help="Top-level category (required)"),
            "Sector": st.column_config.TextColumn(help="Specific segment (required)"),
            "YoY_Growth_%": st.column_config.NumberColumn(format="%.1f", help="Expected yearly growth % (-50 to +100)"),
        }
    )
    if st.button("💾 Save growth table"):
        original_count = len(edited)
        save_growth_table(edited)
        saved = load_growth_table()
        removed = original_count - len(saved)
        if removed > 0:
            st.warning(f"Removed {removed} invalid row(s) (empty fields or growth outside -50% to +100% range).")
        st.success(f"Saved {len(saved)} valid entries.")
        st.rerun()

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
