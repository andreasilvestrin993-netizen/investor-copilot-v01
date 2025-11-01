"""
Configuration management and path constants
"""
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Directory paths
APP_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = (APP_DIR / "data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = (APP_DIR / "output")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Cache file paths
PRICES_CACHE_FILE = CACHE_DIR / "prices_cache.json"
FX_CACHE_FILE = CACHE_DIR / "fx_cache.json"

# CSV file paths
PORTFOLIO_CSV = DATA_DIR / "portfolio.csv"
WATCHLIST_CSV = DATA_DIR / "watchlists.csv"
OVERRIDES_CSV = DATA_DIR / "symbol_overrides.csv"
IND_GROWTH_CSV = DATA_DIR / "industry_growth.csv"
PORTFOLIO_HISTORY_CSV = DATA_DIR / "portfolio_history.csv"

# Base currency
BASE_CCY = "EUR"  # hard-enforced base currency display

# Column definitions
PORTFOLIO_COLS = ["Name", "Symbol", "Quantity", "BEP", "Sector", "Currency"]
WATCHLIST_COLS = ["Name", "Symbol", "Sector", "Currency", "Buy_Low", "Buy_High", 
                  "Target_1Y", "Target_3Y", "Target_5Y", "Target_10Y"]
OVERRIDE_COLS = ["UserSymbol", "ProviderSymbol", "ProviderCurrency"]


def read_yaml(path: Path):
    """
    Read YAML configuration file
    
    Args:
        path: Path to YAML file
    
    Returns:
        dict: Configuration data, empty dict if file not found
    """
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# Load app configuration
CFG = read_yaml(APP_DIR / "config.yaml")
