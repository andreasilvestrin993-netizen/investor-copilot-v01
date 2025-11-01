"""
Cache utilities for daily price and FX rate persistence
"""
import json
from datetime import datetime
from pathlib import Path


# Cache TTL settings (in hours)
PRICES_CACHE_TTL = 24  # 24 hours for prices
FX_CACHE_TTL = 6       # 6 hours for FX rates (more volatile)


def load_daily_cache(cache_file: Path, ttl_hours=24):
    """
    Load cache if it's within TTL, otherwise return empty dict
    
    Args:
        cache_file: Path to cache file
        ttl_hours: Time-to-live in hours (default 24)
    
    Returns:
        dict: Cached data if valid, empty dict otherwise
    """
    if not cache_file.exists():
        return {}
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if cache is within TTL
        cache_timestamp = data.get('timestamp')
        if cache_timestamp:
            cache_time = datetime.fromisoformat(cache_timestamp)
            now = datetime.now()
            hours_diff = (now - cache_time).total_seconds() / 3600
            
            if hours_diff <= ttl_hours:
                return data.get('data', {})
        
        return {}
    except Exception:
        return {}


def save_daily_cache(cache_file: Path, data: dict):
    """
    Save cache with current timestamp
    
    Args:
        cache_file: Path to cache file
        data: Dictionary to cache
    """
    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        pass  # Silent fail for cache save errors
