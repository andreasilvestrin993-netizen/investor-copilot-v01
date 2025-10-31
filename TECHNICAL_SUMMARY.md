# Investor Copilot v01 - Technical Summary

## Overview
Streamlit-based investment portfolio tracker with automated price fetching, multi-currency support, watchlist management, and AI-powered market analysis from YouTube sources.

**Base Currency**: EUR (all displays in euros)  
**Data Provider**: Marketstack API (EOD prices, FX rates)  
**AI Provider**: OpenAI GPT-4 (analysis summaries)

---

## Architecture

### Core Technologies
- **Framework**: Streamlit 1.39.0
- **Data**: Pandas (CSV-based storage)
- **Visualization**: Plotly 5.24.1 (interactive pie charts)
- **APIs**: Marketstack (prices/FX), OpenAI (summaries), YouTubeTranscriptAPI
- **Language**: Python 3.13.9

### File Structure
```
investor-copilot-v01/
├── app.py                          # Main Streamlit app (2740 lines)
├── config.yaml                     # Timezone config
├── requirements.txt                # Dependencies
├── data/
│   ├── portfolio.csv               # Portfolio holdings (37 stocks)
│   ├── watchlists.csv              # Watchlist (202 stocks)
│   ├── symbol_overrides.csv        # Symbol mappings (20 overrides)
│   ├── industry_growth.csv         # Sector YoY growth rates
│   ├── portfolio_history.csv       # Daily snapshots
│   └── cache/
│       ├── prices_cache.json       # Daily price cache
│       └── fx_cache.json           # Daily FX rate cache
└── output/
    └── Analysis/
        └── {month}/
            └── {week}/
                ├── Daily_Summary_{date}.md
                ├── Weekly_Summary_{date}.md
                └── Monthly_Summary_{date}.md
```

---

## Data Layer

### CSV Schema

#### portfolio.csv
| Column   | Type   | Description                    |
|----------|--------|--------------------------------|
| Name     | string | Company name                   |
| Symbol   | string | User ticker symbol             |
| Quantity | float  | Number of shares               |
| BEP      | float  | Break-even price (entry)       |
| Sector   | string | Industry sector                |
| Currency | string | Stock's native currency        |

#### watchlists.csv
| Column     | Type  | Description                      |
|------------|-------|----------------------------------|
| Name       | str   | Company name                     |
| Symbol     | str   | User ticker                      |
| Sector     | str   | Industry sector                  |
| Currency   | str   | Native currency                  |
| Buy_Low    | float | Auto: price × 0.9                |
| Buy_High   | float | Auto: price × 1.1                |
| Target_1Y  | float | Auto: price × (1 + growth)       |
| Target_3Y  | float | Auto: 1Y × (1 + growth)²         |
| Target_5Y  | float | Auto: 1Y × (1 + growth)⁴         |
| Target_10Y | float | Auto: 1Y × (1 + growth)⁹         |

#### symbol_overrides.csv
| Column            | Type   | Description                     |
|-------------------|--------|---------------------------------|
| UserSymbol        | string | Symbol in portfolio/watchlist   |
| ProviderSymbol    | string | Marketstack ticker (e.g. ASML.AS)|
| ProviderCurrency  | string | Currency of provider symbol     |

#### industry_growth.csv
| Column        | Type  | Description               |
|---------------|-------|---------------------------|
| Industry      | str   | Top-level category        |
| Sector        | str   | Specific segment          |
| YoY_Growth_%  | float | Expected annual growth    |

### Supported Currencies
EUR, USD, GBP, CHF, CAD, SEK, DKK, PLN, HKD, JPY, AUD, NOK, GBX

---

## Core Functions

### 1. Caching System (Daily Persistence)
**Files**: `data/cache/prices_cache.json`, `data/cache/fx_cache.json`

```python
load_daily_cache(cache_file: Path) -> dict
save_daily_cache(cache_file: Path, data: dict)
```
- Loads cache if date == today, otherwise empty dict
- First app open of day: fetches from Marketstack
- Subsequent reloads: uses cached data (no API calls)
- Next day: auto-expires, fresh fetch on first load

### 2. Price Fetching
```python
fetch_eod_prices(symbols, marketstack_key) -> dict[str, float]
```
- Batch API calls (80 symbols max per request)
- Checks daily cache first
- Returns `{ProviderSymbol: close_price}`
- Saves to cache after fetch

### 3. FX Rate Conversion
```python
fetch_fx_map_eur(marketstack_key) -> dict[str, float]
```
**Triple Fallback**:
1. **Marketstack**: EURUSD, EURGBP, EURCHF latest EOD
2. **Frankfurter API**: ECB rates if Marketstack fails
3. **Emergency Defaults**: USD=0.92, GBP=0.85, CHF=1.05

**Derived Rates**:
- HKD = USD / 7.8 (USD peg)
- CAD = USD / 1.35
- SEK = EUR / 11.5
- DKK = EUR / 7.46
- PLN = EUR / 4.35

```python
to_eur(amount, ccy, fx_map) -> float
```
Converts any amount to EUR using FX map.

### 4. Symbol Resolution
```python
resolve_provider_symbol(user_symbol, name_hint, ccy, marketstack_key, prices_cache, auto_save) -> tuple[str, str]
```
**Resolution Order**:
1. Check `symbol_overrides.csv` (manual mappings)
2. Check `prices_cache` (previously resolved)
3. Search Marketstack API by name hint
4. Return (ProviderSymbol, ProviderCurrency)

**Auto-save**: Optionally saves successful resolutions to overrides

### 5. Company Search
```python
search_companies(query, marketstack_key, limit=10) -> list[dict]
```
Autocomplete search returning:
- Company name
- Symbol
- Exchange
- Sector (if available)
- Currency

### 6. CSV Import Intelligence
```python
detect_csv_format(file_content) -> tuple[str, str]
read_csv_smart(uploaded_file) -> tuple[DataFrame, str, str]
show_column_mapping_ui(uploaded_df, expected_cols, csv_type) -> DataFrame
```
**Detects**:
- Delimiter: semicolon (European) vs comma (US)
- Decimal: comma vs period
- Auto-maps columns if headers don't match

---

## UI Structure (5 Tabs)

### Tab 1: Dashboard 🏠
**Displays**:
- Total portfolio value (EUR)
- Total P&L (EUR and %)
- Portfolio composition pie chart (Top 10 + Other)
- Sector/Industry breakdown pie chart (Top 10 + Other)
- Portfolio value over time (line chart from daily snapshots)
- Watchlist opportunities (stocks near buy targets)

**Pie Chart Features**:
- Smart color coding by industry category
  - Blues: Technology (AI, Cloud, Semiconductors)
  - Grays: Industrials (Aerospace, Manufacturing)
  - Greens: Healthcare (Biotech, Genomics)
  - Yellows: Financial
  - Oranges: Consumer/E-commerce
  - Reds: Automotive/Mobility
  - Purples: Energy
- Hover: Stock name, €value, percentage
- Sector chart: Shows industry name, full details on hover

**Data Flow**:
1. Pre-process: Build `symbol_map` (user → provider symbols with overrides)
2. Batch fetch: Get all prices in one Marketstack call
3. Calculate: Position values, P&L, percentages
4. Aggregate: Group for Top 10 + Other

### Tab 2: Portfolio 💼
**Features**:
- Add/edit positions with autocomplete
- Auto-populate sector from search
- CSV import/export with smart column mapping
- Delete positions
- Refresh prices button (clears cache, forces fresh fetch)
- Sortable columns (click headers)

**Columns Displayed**:
- Name, Symbol, Quantity, BEP, Sector, Currency
- Current Price (€), Total Value (€)
- P&L (€), P&L (%)

**Actions**:
- 🔄 Refresh Prices: Clears in-memory + disk cache, forces fresh Marketstack fetch
- 🗑️ Clear Portfolio: Removes all positions
- 💾 Save Snapshot: Appends current portfolio value to history

### Tab 3: Watchlists 🔭
**Features**:
- Add stocks with autocomplete
- Auto-calculate buy ranges (±10% of current price)
- Auto-calculate targets from sector growth rates
- Edit targets manually
- Filter by sector, currency, search text
- Sortable columns (st.dataframe)
- Add symbol overrides for missing prices

**Auto-Calculations**:
- **Buy Low**: Current price × 0.9
- **Buy High**: Current price × 1.1
- **Target 1Y**: Current price × (1 + sector_growth_rate)
- **Target 3Y**: Target_1Y × (1 + growth)²
- **Target 5Y**: Target_1Y × (1 + growth)⁴
- **Target 10Y**: Target_1Y × (1 + growth)⁹

**Columns Displayed**:
- Name, Symbol, Sector, Currency
- Current Price (€), Buy Low (€), Buy High (€)
- % from Buy Low, % from Buy High
- Target 1Y/3Y/5Y/10Y (€)
- Upside to 1Y/3Y/5Y/10Y (%)

### Tab 4: Analysis 📰
**Purpose**: Generate AI summaries from financial YouTube channels

**Workflow**:
1. Input YouTube channel URLs or @handles
2. Fetch latest videos via RSS feed
3. Download transcripts via YouTubeTranscriptAPI
4. Send to OpenAI GPT-4 for summarization
5. Save as Markdown + DOCX in `output/Analysis/{month}/{week}/`

**Summary Types**:
- **SEED**: Baseline from last 3 videos per channel
- **DAILY**: Today's videos summarized
- **WEEKLY**: Roll-up of daily summaries
- **MONTHLY**: Roll-up of weekly summaries

**Storage Structure**:
```
output/Analysis/
└── 10.2025/                    # Month folder
    └── 28-01 November 2025/    # Week folder
        ├── Daily_Summary_30.10.2025.md
        ├── Daily_Summary_30.10.2025.docx
        ├── Weekly_Summary_01.11.2025.md
        └── Monthly_Summary_30.11.2025.md
```

**Browse Feature**: Navigate month → week → view/download summaries

### Tab 5: Settings ⚙️
**Editable Tables**:
1. **Industry Growth Rates**: Manage sector YoY growth expectations
2. **Symbol Overrides**: Manual symbol → provider mappings
3. **Preferences**: Fixed to EUR, configurable theme

---

## Key Algorithms

### 1. Symbol Pre-Processing (Dashboard & Watchlist)
```python
# Build symbol map BEFORE fetching prices
symbol_map = {}
for symbol in portfolio_symbols:
    provider_sym, provider_ccy = resolve_provider_symbol(
        user_symbol=symbol,
        name_hint=company_name,
        ccy=user_currency,
        marketstack_key=API_KEY,
        prices_cache={}
    )
    symbol_map[symbol] = provider_sym

# Batch fetch all prices
prices = fetch_eod_prices(list(symbol_map.values()), API_KEY)

# Calculate values
for row in portfolio:
    provider_sym = symbol_map[row['Symbol']]
    price = prices.get(provider_sym)
    value_native = row['Quantity'] * price
    value_eur = to_eur(value_native, provider_ccy, fx_map)
```

### 2. Top 10 + Other Aggregation
```python
# Sort by value descending
df_sorted = df.sort_values("Value", ascending=False)

if len(df_sorted) > 10:
    top_10 = df_sorted.head(10)
    other_value = df_sorted.iloc[10:]["Value"].sum()
    other_pct = df_sorted.iloc[10:]["% of Portfolio"].sum()
    
    other_row = pd.DataFrame([{
        "Stock": "Other",
        "Value": other_value,
        "% of Portfolio": other_pct
    }])
    df_final = pd.concat([top_10, other_row])
else:
    df_final = df_sorted
```

### 3. Smart Sector Color Coding
```python
def get_sector_color(sector_name):
    sector_lower = sector_name.lower()
    
    if 'ai' in sector_lower or 'machine learning' in sector_lower:
        return '#357ABD'  # Medium blue
    elif 'semiconductor' in sector_lower:
        return '#2E5F8F'  # Dark blue
    elif 'aerospace' in sector_lower or 'defense' in sector_lower:
        return '#7F8C8D'  # Dark gray
    elif 'biotech' in sector_lower or 'genomic' in sector_lower:
        return '#27AE60'  # Emerald
    # ... 20+ category mappings
    else:
        return '#16A085'  # Teal (default)
```

### 4. Industry Extraction (Sector Pie Chart)
```python
# Extract industry name (before parentheses/dash)
industry = sector.split('(')[0].split('—')[0].strip()

# Group by industry, aggregate values
industry_df = sector_df.groupby("Industry").agg({
    "Value": "sum",
    "% of Portfolio": "sum",
    "Full_Sector": lambda x: "<br>".join(x)  # Keep details for hover
})
```

---

## Performance Optimizations

### 1. Daily Caching
- **First load of day**: ~5s (Marketstack API calls)
- **Subsequent reloads same day**: <1s (cached data)
- Cache files: JSON with date stamps in `data/cache/`

### 2. Batch API Calls
- Fetches 80 symbols per Marketstack request (instead of 1-by-1)
- Portfolio (37 stocks) + Watchlist (202 stocks) = ~3 API calls total

### 3. Symbol Resolution Caching
- Stores resolved symbols in `symbol_overrides.csv`
- Next run skips search API calls for known symbols

### 4. Session State
- In-memory cache for current session
- Avoids re-reading CSV files on every interaction

---

## Data Flows

### Portfolio Value Calculation
```
User Symbol (e.g., "ASML")
    ↓
Check symbol_overrides.csv → ASML.AS (EUR)
    ↓
Fetch price from Marketstack → €645.30
    ↓
Calculate: Quantity × Price = 10 × €645.30 = €6,453.00
    ↓
Convert to EUR (already EUR) → €6,453.00
    ↓
Compare to BEP: (€645.30 - €600.00) / €600.00 = +7.55%
    ↓
Sum all positions → Total Portfolio Value
```

### Watchlist Target Calculation
```
Symbol: NVDA (USD)
    ↓
Fetch current price → $140.00
    ↓
Convert to EUR: $140 × 0.8658 (USD→EUR) = €121.21
    ↓
Lookup sector: "Semiconductors" → YoY Growth = 22%
    ↓
Auto-calculate targets:
  • Buy Low:  €121.21 × 0.9 = €109.09
  • Buy High: €121.21 × 1.1 = €133.33
  • Target 1Y: €121.21 × 1.22 = €147.88
  • Target 3Y: €147.88 × 1.22² = €220.05
  • Target 5Y: €147.88 × 1.22⁴ = €388.84
    ↓
Calculate upside: (€147.88 - €121.21) / €121.21 = +22%
```

### Analysis Summary Generation
```
YouTube Channel URL (@StocksToday)
    ↓
Resolve to channel ID via API
    ↓
Fetch RSS feed → Get latest 3 videos
    ↓
Extract video IDs
    ↓
Fetch transcripts via YouTubeTranscriptAPI
    ↓
Combine transcripts (10,000+ words)
    ↓
Send to OpenAI GPT-4 with prompt:
  "Summarize key investment insights, sentiment, risks..."
    ↓
Receive summary (~500 words)
    ↓
Save to output/Analysis/10.2025/28-01 November/
  • Seed_Summary_30.10.2025.md
  • Seed_Summary_30.10.2025.docx
```

---

## Error Handling

### Price Fetch Failures
- Returns `None` for missing prices
- UI displays "-" instead of crashing
- User can add manual override in Settings

### FX Rate Failures
- Triple fallback (Marketstack → Frankfurter → Emergency defaults)
- Never blocks app startup

### Symbol Resolution
- Tries override → cache → API search
- If all fail, shows symbol as-is with missing price

### API Rate Limits
- Batches requests (80 symbols max)
- 0.2s sleep between batches
- Daily caching reduces API usage by ~95%

### CSV Import Errors
- Smart format detection (European vs US)
- Column mapping UI if headers don't match
- Validates numeric fields, coerces to float

---

## Configuration

### Environment Variables (.env)
```bash
MARKETSTACK_KEY=your_api_key
OPENAI_API_KEY=your_openai_key
```

### config.yaml
```yaml
app:
  timezone: Europe/Madrid
```

---

## Dependencies (requirements.txt)
```
streamlit==1.39.0
pandas
requests
plotly==5.24.1
youtube-transcript-api
python-docx
python-dotenv
pyyaml
openai
```

---

## API Usage Summary

### Marketstack
- **Endpoint**: `/v1/eod/latest`
- **Usage**: EOD prices for all symbols
- **Rate**: ~3 calls per app load (portfolio + watchlist)
- **Cache**: 24 hours (daily refresh)

### Marketstack FX
- **Endpoint**: `/v1/tickers/{EURUSD|EURGBP|EURCHF}/eod/latest`
- **Usage**: FX rates to EUR
- **Rate**: 3 calls per day (if cache expired)
- **Fallback**: Frankfurter API

### OpenAI
- **Model**: GPT-4
- **Usage**: Summary generation (manual trigger only)
- **Prompt**: ~10,000 tokens (transcripts) → ~500 tokens (summary)
- **Cost**: ~$0.30 per summary

### YouTube
- **YouTubeTranscriptAPI**: Free, no key required
- **RSS Feeds**: Free, no rate limit
- **Limitations**: Transcripts must be enabled by channel owner

---

## Current State (as of Oct 31, 2025)

### Portfolio
- **37 stocks** tracked
- **Total value**: €104.7K
- **Currencies**: EUR, USD, GBP, HKD
- **Top holdings**: NVIDIA, Palantir, Tesla, Microsoft, etc.

### Watchlist
- **202 stocks** tracked (down from 251)
- **20 symbol overrides** active
- **Sectors**: 50+ across Technology, Energy, Healthcare, Industrials

### Cache Status
- Daily price cache: Active
- Daily FX cache: Active
- Symbol resolution: 20 pre-mapped

---

## Future Enhancement Ideas
1. Real-time price updates (WebSocket)
2. Mobile-responsive UI
3. Push notifications for price alerts
4. Advanced charting (candlesticks, indicators)
5. Multi-portfolio support
6. Tax reporting (realized gains)
7. Dividend tracking
8. Correlation analysis
9. Risk metrics (Sharpe, beta, volatility)
10. Export to PDF/Excel

---

## Development Notes

### Code Quality
- **Lines**: 2,740 in app.py
- **Functions**: 41 core functions
- **No classes**: Functional programming style
- **Type hints**: Partial (modern Python 3.13)

### Testing
- Manual testing via Streamlit UI
- No automated tests currently

### Version Control
- Git repository: `investor-copilot-v01`
- GitHub: `andreasilvestrin993-netizen/investor-copilot-v01`
- Recent commits: Daily caching, watchlist cleanup, UI fixes

### Known Issues
- None reported (all critical bugs fixed)

---

**Document Version**: 1.0  
**Last Updated**: October 31, 2025  
**Author**: AI Assistant (GitHub Copilot)
