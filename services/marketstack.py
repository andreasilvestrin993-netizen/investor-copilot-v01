"""
Marketstack API service for fetching EOD prices, FX rates, and symbol search
"""
import requests
import streamlit as st
from datetime import datetime
import time
import pandas as pd

from config.settings import (
    DATA_DIR, 
    FX_CACHE_FILE, 
    PRICES_CACHE_FILE, 
    OVERRIDES_CSV, 
    OVERRIDE_COLS
)
from utils.cache import save_daily_cache
from utils.sector_utils import map_industry_to_sector
from utils.csv_utils import load_csv, save_csv

def ms_get(url, params):
    """Generic helper for Marketstack GET with query params"""
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
    Cached for 6 hours. Uses Frankfurter API as fallback.
    """
    # Check if cache is still valid (6h TTL)
    now = datetime.now()
    cache_is_valid = (st.session_state.last_fx_fetch and 
                     (now - st.session_state.last_fx_fetch).total_seconds() < 6 * 3600)
    
    if cache_is_valid and st.session_state.fx_cache:
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
    """Load symbol overrides from CSV"""
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

def resolve_provider_symbol(user_symbol: str, name_hint: str, ccy: str, marketstack_key: str, prices_cache: dict, auto_save: bool = False) -> tuple[str, str]:
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
    
    Uses live Marketstack rates (cached 6h) with Frankfurter fallback.
    All portfolio calculations use this single function to ensure consistency.
    """
    if not ccy or ccy == "EUR":
        return float(amount)
    rate = fx_map.get(ccy, 1.0)
    return float(amount) * rate
