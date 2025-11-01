"""
Formatting utilities for displaying numbers, percentages, and currency values
"""
import pandas as pd
import math

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

def clamp_format_number(x):
    """Format large numbers with B/M/K suffixes (no currency symbol)"""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    n = float(x)
    absn = abs(n)
    if absn >= 1_000_000_000:
        return f"{n/1_000_000_000:.0f}B"
    if absn >= 1_000_000:
        return f"{n/1_000_000:.0f}M"
    if absn >= 1_000:
        return f"{n/1_000:.0f}K"
    return f"{int(round(n, 0))}"

def fmt_pct(x):
    """Format percentage to 1 decimal (alternative name)"""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.1f}%"
