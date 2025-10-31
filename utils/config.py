"""
Configuration and constants for the investor copilot app
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Directories
APP_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = (APP_DIR / "data")
OUTPUT_DIR = (APP_DIR / "output")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Files
IND_GROWTH_CSV = DATA_DIR / "industry_growth.csv"
PORTFOLIO_CSV = DATA_DIR / "portfolio.csv"
WATCHLIST_CSV = DATA_DIR / "watchlists.csv"
OVERRIDES_CSV = DATA_DIR / "symbol_overrides.csv"

# Column definitions
PORTFOLIO_COLS = ["Name","Symbol","Quantity","BEP","Sector","Currency"]
WATCHLIST_COLS = ["Name","Symbol","Sector","Currency","Buy_Low","Buy_High","Target_1Y","Target_3Y","Target_5Y","Target_10Y"]
OVERRIDE_COLS = ["UserSymbol","ProviderSymbol","ProviderCurrency"]

# Base currency
BASE_CCY = "EUR"

# API Keys
MARKETSTACK_KEY = os.getenv("MARKETSTACK_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

def read_yaml(path: Path):
    """Read YAML configuration file"""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

# Load config
CFG = read_yaml(APP_DIR / "config.yaml")

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'symbol_cache' not in st.session_state:
        st.session_state.symbol_cache = {}
    if 'prices_cache' not in st.session_state:
        st.session_state.prices_cache = {}
    if 'last_price_fetch' not in st.session_state:
        st.session_state.last_price_fetch = None
    if 'pending_upload' not in st.session_state:
        st.session_state.pending_upload = None
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    if 'upload_type' not in st.session_state:
        st.session_state.upload_type = None
    if 'csv_format_info' not in st.session_state:
        st.session_state.csv_format_info = None
