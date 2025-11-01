"""
Sector and industry utilities for mapping and growth table management
"""
import pandas as pd
from pathlib import Path

# Get paths from config
from config.settings import DATA_DIR

IND_GROWTH_CSV = DATA_DIR / "industry_growth.csv"

# Map industry/sector names to simplified categories
SECTOR_MAPPING = {
    "Technology": ["Technology", "AI & Machine Learning", "Cloud Infrastructure", "Software", "Semiconductors"],
    "Energy & Clean Tech": ["Energy", "Renewable", "Solar", "Wind", "Nuclear"],
    "Automotive & Mobility": ["Automotive", "EV", "Electric Vehicle", "Mobility"],
    "Financial Services": ["Finance", "Fintech", "Banking", "Insurance"],
    "Healthcare & Life Sciences": ["Healthcare", "Biotech", "Pharma", "Medical"],
    "Consumer Tech & Digital Media": ["Consumer", "E-Commerce", "Media", "Gaming"],
    "Telecom & Connectivity": ["Telecom", "Communication", "5G"],
    "Industrials": ["Industrial", "Manufacturing", "Aerospace"],
    "Materials & Mining": ["Materials", "Mining", "Metals"],
    "Real Estate & Infrastructure": ["Real Estate", "Infrastructure", "Construction"]
}

def get_sector_list():
    """Get list of all sectors from industry_growth.csv"""
    if not IND_GROWTH_CSV.exists():
        return []
    growth_df = pd.read_csv(IND_GROWTH_CSV)
    sectors = sorted(growth_df["Sector"].unique().tolist())
    return sectors

def map_industry_to_sector(industry_text):
    """Map Marketstack industry to our sector list"""
    if not industry_text:
        return None
    
    industry_lower = industry_text.lower()
    
    # Direct match first
    sectors = get_sector_list()
    for sector in sectors:
        if sector.lower() in industry_lower or industry_lower in sector.lower():
            return sector
    
    # Keyword mapping
    for sector, keywords in SECTOR_MAPPING.items():
        for keyword in keywords:
            if keyword.lower() in industry_lower:
                return sector
    
    return None

def load_growth_table():
    """Load industry growth table from CSV"""
    return pd.read_csv(IND_GROWTH_CSV)[["Industry", "Sector", "YoY_Growth_%"]]

def save_growth_table(df):
    """Save industry growth table with validation rules"""
    # Validation: Remove rows with any empty values
    df_clean = df.dropna(subset=["Industry", "Sector", "YoY_Growth_%"])
    
    # Validation: Remove rows where any field is empty string
    df_clean = df_clean[
        (df_clean["Industry"].str.strip() != "") & 
        (df_clean["Sector"].str.strip() != "") &
        (df_clean["YoY_Growth_%"].notna())
    ]
    
    # Validation: Ensure growth is numeric and reasonable (-50 to +100%)
    df_clean["YoY_Growth_%"] = pd.to_numeric(df_clean["YoY_Growth_%"], errors='coerce')
    df_clean = df_clean[
        (df_clean["YoY_Growth_%"] >= -50) & 
        (df_clean["YoY_Growth_%"] <= 100)
    ]
    
    df_clean[["Industry", "Sector", "YoY_Growth_%"]].to_csv(IND_GROWTH_CSV, index=False)

def init_industry_growth_csv():
    """Initialize industry growth CSV with seed data if it doesn't exist"""
    if not IND_GROWTH_CSV.exists():
        seed = pd.DataFrame([
            # Technology (≤10 industries total, ≤6 sectors each across whole app)
            ["Technology", "AI & Machine Learning", 30.0],
            ["Technology", "Cloud Infrastructure & Services", 27.0],
            ["Technology", "Cybersecurity", 15.0],
            ["Technology", "Software (Applications & DevOps)", 12.5],
            ["Technology", "Semiconductors", 10.5],
            ["Technology", "Data Infrastructure (Storage, Networking, CDNs)", 23.0],
            # Energy & Clean Tech
            ["Energy & Clean Tech", "EV & Battery Systems", 12.0],
            ["Energy & Clean Tech", "Solar & Wind Power", 11.0],
            ["Energy & Clean Tech", "Hydrogen & Fuel Cells", 8.0],
            ["Energy & Clean Tech", "Nuclear Energy (incl. SMRs, Uranium)", 9.0],
            ["Energy & Clean Tech", "Utilities & Grid Tech", 5.0],
            ["Energy & Clean Tech", "Renewable Infrastructure", 8.5],
            # Industrials
            ["Industrials", "Aerospace & Defense", 11.0],
            ["Industrials", "Advanced Manufacturing & Equipment", 7.5],
            ["Industrials", "Construction & Engineering", 4.5],
            ["Industrials", "Security & Infrastructure", 6.5],
            ["Industrials", "Water & Environmental Systems", 5.0],
            # Automotive & Mobility
            ["Automotive & Mobility", "EV Manufacturers", 11.0],
            ["Automotive & Mobility", "Autonomous Vehicles & Lidar", 14.0],
            ["Automotive & Mobility", "Urban Air Mobility (eVTOL, Drones)", 15.0],
            ["Automotive & Mobility", "Telematics & Mobility Tech", 10.0],
            # Financial Services
            ["Financial Services", "Fintech & Neo-Banks", 10.0],
            ["Financial Services", "Trading Platforms & Exchanges", 8.0],
            ["Financial Services", "Crypto Infrastructure", 20.0],
            ["Financial Services", "AI-based Financial Services", 15.0],
            # Healthcare & Life Sciences
            ["Healthcare & Life Sciences", "Biotechnology & Genomics", 12.0],
            ["Healthcare & Life Sciences", "Medical Devices & Diagnostics", 8.5],
            ["Healthcare & Life Sciences", "Healthcare Software & Analytics", 10.0],
            ["Healthcare & Life Sciences", "Pharmaceuticals", 6.0],
            # Consumer Tech & Digital Media
            ["Consumer Tech & Digital Media", "E-Commerce Platforms", 9.5],
            ["Consumer Tech & Digital Media", "Social Media & Content", 4.0],
            ["Consumer Tech & Digital Media", "Gaming & Interactive Media", 5.0],
            ["Consumer Tech & Digital Media", "Consumer Electronics", 4.5],
            # Telecom & Connectivity
            ["Telecom & Connectivity", "Communication Infrastructure", 5.5],
            ["Telecom & Connectivity", "Networking & 5G Tech", 8.5],
            ["Telecom & Connectivity", "Telecom Services", 4.5],
            # Materials & Mining
            ["Materials & Mining", "Lithium & Battery Materials", 9.0],
            ["Materials & Mining", "Rare Earths & Advanced Materials", 7.0],
            ["Materials & Mining", "Uranium & Nuclear Fuel Supply", 8.5],
            # Real Estate & Infrastructure
            ["Real Estate & Infrastructure", "Data Centers", 10.0],
            ["Real Estate & Infrastructure", "Smart Infrastructure", 7.5],
            ["Real Estate & Infrastructure", "Real Estate Tech & Services", 5.5],
        ], columns=["Industry", "Sector", "YoY_Growth_%"])
        seed.to_csv(IND_GROWTH_CSV, index=False)
