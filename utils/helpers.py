"""
General helper utilities
"""
import os
from datetime import date
import datetime as dt
import math

def get_secret(name, default=""):
    """Get environment variable secret"""
    return os.getenv(name, default)

def fmt_price(x):
    """Format price to 2 decimals"""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.2f}"

def guess_week_folder(dt_obj: date):
    """Generate week folder name (Monday-Friday format)"""
    monday = dt_obj - dt.timedelta(days=dt_obj.weekday())
    friday = monday + dt.timedelta(days=4)
    return f"{monday.day:02d}-{friday.day:02d} {friday.strftime('%B')} {friday.year}"

def month_folder(dt_obj: date):
    """Generate month folder name (MM.YYYY format)"""
    return f"{dt_obj.strftime('%m.%Y')}"
