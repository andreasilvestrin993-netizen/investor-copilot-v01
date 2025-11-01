"""
Financial calculations: EAGR targets, P/L, formatting
"""
import pandas as pd


def calc_eagr_targets(current_price, target_1y, sector_growth_pct, r52w_pct=None):
    """
    Calculate multi-year targets using EAGR (Enhanced Annual Growth Rate).
    Blends sector growth (70%) with 52-week momentum (30%).
    
    Args:
        current_price: Current stock price
        target_1y: User's 1-year target (can be manually adjusted)
        sector_growth_pct: Sector YoY growth rate (as percentage, e.g., 15.0 for 15%)
        r52w_pct: 52-week price change (as percentage, optional)
    
    Returns:
        tuple: (target_1y_adjusted, target_3y, target_5y, target_10y)
    """
    if not current_price or current_price <= 0:
        return None, None, None, None
    
    if not target_1y or target_1y <= 0:
        return None, None, None, None
    
    # If no 52W momentum data, use sector growth
    if r52w_pct is None:
        r52w_pct = sector_growth_pct
    
    # Clamp 52W momentum to reasonable range (-50% to +50%)
    r52w_pct = max(min(r52w_pct, 50), -50)
    
    # Calculate EAGR: 70% sector growth + 30% 52-week momentum
    eagr_pct = (0.7 * sector_growth_pct) + (0.3 * r52w_pct)
    
    # Bound EAGR to -10% to +25%
    eagr_pct = max(min(eagr_pct, 25), -10)
    eagr = eagr_pct / 100  # Convert to decimal
    
    # Calculate 1Y growth implied by user's target
    g1 = (target_1y / current_price) - 1
    
    # Smooth the 1Y growth: 60% user target + 40% EAGR
    g1_smooth = (0.6 * g1) + (0.4 * eagr)
    
    # Adjusted 1Y target based on smoothing
    t1_final = round(current_price * (1 + g1_smooth), 2)
    
    # Multi-year targets using EAGR compounding
    t3 = round(t1_final * ((1 + eagr) ** 2), 2)   # 2 years after year 1
    t5 = round(t1_final * ((1 + eagr) ** 4), 2)   # 4 years after year 1
    t10 = round(t1_final * ((1 + eagr) ** 9), 2)  # 9 years after year 1
    
    return t1_final, t3, t5, t10


def format_price(value, currency="EUR"):
    """Format price to 2 decimals with currency symbol"""
    if value is None or pd.isna(value):
        return "-"
    try:
        v = float(value)
        if currency == "EUR":
            return f"€{v:,.2f}"
        elif currency == "USD":
            return f"${v:,.2f}"
        elif currency == "GBP":
            return f"£{v:,.2f}"
        else:
            return f"{v:,.2f} {currency}"
    except:
        return "-"


def format_percent(value):
    """Format percentage value"""
    if value is None or pd.isna(value):
        return "-"
    try:
        v = float(value)
        return f"{v:+.2f}%"
    except:
        return "-"


def format_quantity(value):
    """Format quantity (whole number or decimal)"""
    if value is None or pd.isna(value):
        return "-"
    try:
        v = float(value)
        if v == int(v):
            return str(int(v))
        return f"{v:.3f}"
    except:
        return "-"
