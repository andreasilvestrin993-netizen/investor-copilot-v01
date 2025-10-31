"""
Formatting utilities for displaying data
"""
import pandas as pd

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
