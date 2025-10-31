"""
Portfolio history tracking for daily snapshots
"""
import pandas as pd
from datetime import datetime
from pathlib import Path

def save_daily_snapshot(portfolio_value, pnl_eur, pnl_pct, data_dir):
    """Save daily portfolio snapshot"""
    history_file = data_dir / "portfolio_history.csv"
    today = datetime.now().date().isoformat()
    
    # Load existing history
    if history_file.exists():
        try:
            history = pd.read_csv(history_file, sep=';')
        except:
            history = pd.DataFrame(columns=["Date", "Value_EUR", "PnL_EUR", "PnL_Pct"])
    else:
        history = pd.DataFrame(columns=["Date", "Value_EUR", "PnL_EUR", "PnL_Pct"])
    
    # Check if today already exists
    if not history.empty and today in history["Date"].values:
        # Update today's values
        history.loc[history["Date"] == today, ["Value_EUR", "PnL_EUR", "PnL_Pct"]] = [portfolio_value, pnl_eur, pnl_pct]
    else:
        # Add new row
        new_row = pd.DataFrame([{
            "Date": today,
            "Value_EUR": portfolio_value,
            "PnL_EUR": pnl_eur,
            "PnL_Pct": pnl_pct
        }])
        history = pd.concat([history, new_row], ignore_index=True)
    
    # Save back
    history.to_csv(history_file, index=False, sep=';')
    return history

def load_portfolio_history(data_dir):
    """Load portfolio history"""
    history_file = data_dir / "portfolio_history.csv"
    if history_file.exists():
        try:
            return pd.read_csv(history_file, sep=';')
        except:
            return pd.DataFrame(columns=["Date", "Value_EUR", "PnL_EUR", "PnL_Pct"])
    return pd.DataFrame(columns=["Date", "Value_EUR", "PnL_EUR", "PnL_Pct"])
