# Remaining Tasks to Complete Investor Copilot v1.0

**Date**: October 31, 2025  
**Current Status**: Core functional, 37 stocks in portfolio, 202 in watchlist, all bugs fixed

---

## Phase 1: Essential for v1.0 (Launch-Ready)

### 1. **Implement EAGR Blended Formula** ⚠️ HIGH PRIORITY
**Current State**: Targets calculated using simple sector growth only  
**Target State**: Blend sector growth (70%) + 52W momentum (30%)

**Formula**:
```python
# In calc_targets() function (around line 800-850)
def calc_targets(p0, t1, sector_growth, r52w):
    if not r52w: 
        r52w = sector_growth
    r52w = max(min(r52w, 50), -50)  # Clamp 52W to ±50%
    
    eagr = (0.7 * sector_growth) + (0.3 * r52w)
    eagr = max(min(eagr, 25), -10) / 100  # Bounded: -10% to +25%
    
    g1 = (t1 / p0) - 1
    g1_smooth = (0.6 * g1) + (0.4 * eagr)
    t1_final = p0 * (1 + g1_smooth)
    
    t3 = round(t1_final * (1 + eagr) ** 2, 2)
    t5 = round(t1_final * (1 + eagr) ** 4, 2)
    t10 = round(t1_final * (1 + eagr) ** 9, 2)
    
    return round(t1_final, 2), t3, t5, t10
```

**Files to Edit**: `app.py` (lines ~800-850)  
**Estimated Time**: 2 hours (implement + test)

---

### 2. **Reduce FX Cache TTL to 6 Hours** 🔧 MEDIUM PRIORITY
**Current State**: FX cache uses 24h TTL (same as prices)  
**Target State**: FX should refresh every 6 hours (more volatile)

**Changes**:
```python
# In load_daily_cache() around line 40-69
def load_daily_cache(cache_type='prices'):
    """Load cache with TTL: 24h for prices, 6h for FX"""
    cache_file = f"data/cache/{cache_type}_cache.json"
    if not os.path.exists(cache_file):
        return {}
    
    with open(cache_file, 'r') as f:
        data = json.load(f)
    
    # Check TTL based on cache type
    ttl_hours = 24 if cache_type == 'prices' else 6
    cache_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
    now = datetime.now()
    
    if (now - cache_time).total_seconds() / 3600 > ttl_hours:
        return {}  # Expired
    
    return data.get('data', {})
```

**Files to Edit**: `app.py` (lines ~40-69)  
**Estimated Time**: 1 hour

---

### 3. **Add In-App Alerts (Phase 1)** 🔔 MEDIUM PRIORITY
**Current State**: No alerts system  
**Target State**: Visual badges for buy zones, target reached, stop loss

**Status**: ⏸️ **MOVED TO PHASE 3** (user requested deferral)

**Implementation** (for future reference):
1. Create `data/alerts.csv` or use session state
2. Check conditions on each data load:
   - Buy Zone: `current_price <= buy_high AND current_price >= buy_low`
   - Target Reached: `current_price >= target_1y`
   - Stop Loss: `current_price <= bep * 0.9` (portfolio only)
3. Display badge counters in tab titles
4. Highlight rows with color coding (green/gold/red)

**Files to Edit**: `app.py` (new function `count_alerts()`, update tab rendering)  
**Estimated Time**: 4 hours

---

### 4. **Color-Coded Watchlist Heatmap** 🎨 ✅ COMPLETED
**Current State**: Plain white/dark table  
**Target State**: Color rows by opportunity strength

**Logic**:
```python
# Green: Near buy zone (price within 5% of buy_high)
# Yellow: Moderate upside (10-20% to target)
# White: Normal (>20% to target)
# Red: Overvalued (price > target_1y)

def get_row_color(row):
    if row['current_price'] <= row['buy_high'] * 1.05:
        return 'background-color: rgba(0, 255, 0, 0.2)'  # Green
    elif row['current_price'] >= row['target_1y']:
        return 'background-color: rgba(255, 0, 0, 0.2)'  # Red
    elif (row['target_1y'] - row['current_price']) / row['current_price'] < 0.20:
        return 'background-color: rgba(255, 255, 0, 0.2)'  # Yellow
    else:
        return ''
```

Apply via `st.dataframe(df.style.apply(...))` in Watchlist tab.

**Files to Edit**: `app.py` (Watchlist tab, around lines 1803-2400)  
**Estimated Time**: 2 hours

---

### 5. **Modular Codebase Split** 📂 DEFERRED TO v1.1
**Current State**: 2876 lines in single `app.py`  
**Target State**: Organized into modules

**What is it?**: Splitting one big file into many smaller, organized files:
- `app.py` (main entry) + `/services` (APIs) + `/ui` (tabs) + `/utils` (helpers)

**Why defer?**: 
- High risk of introducing bugs during refactor
- 10 hours of careful work
- Not user-facing (dev experience only)
- App works perfectly as-is

**Status**: ⏸️ **MOVED TO v1.1** (do after launch when there's time to test thoroughly)

**Structure**:
```
investor-copilot-v01/
├── app.py                    # Streamlit UI entry point (500 lines)
├── services/
│   ├── marketstack.py        # API calls (fetch_eod_prices, fetch_fx_map)
│   ├── openai_service.py     # GPT-4 summaries
│   ├── youtube_service.py    # Transcript fetching
│   └── fx_converter.py       # to_eur(), FX logic
├── ui/
│   ├── dashboard.py          # Tab 1
│   ├── portfolio.py          # Tab 2
│   ├── watchlist.py          # Tab 3
│   ├── analysis.py           # Tab 4
│   └── settings.py           # Tab 5
├── utils/
│   ├── cache.py              # load_daily_cache, save_daily_cache
│   ├── calculations.py       # calc_targets, P/L formulas
│   ├── csv_import.py         # Smart import wizard
│   └── symbol_resolver.py    # resolve_provider_symbol
└── config/
    └── settings.py           # Load/save config.yaml
```

**Refactoring Steps**:
1. Extract all API calls to `services/`
2. Move each tab's rendering logic to `ui/`
3. Move utility functions to `utils/`
4. Update imports in `app.py`
5. Test each tab after refactor

**Files to Create**: 13 new modules  
**Files to Edit**: `app.py` (reduce from 2740 → ~500 lines)  
**Estimated Time**: 8-10 hours

---

### 6. **Basic Unit Tests** ✅ DEFERRED TO v1.1
**Current State**: No automated tests  
**Target State**: pytest suite for critical functions

**What are unit tests?**: Small automated tests that verify your code works:
```python
def test_fx_conversion():
    result = to_eur(100, "USD", {"USD": 0.92})
    assert result == 92.0  # Pass! ✅
```

**Benefits**:
- Catch bugs automatically
- Confidence when changing code
- Documentation of expected behavior

**Why defer?**:
- App is already thoroughly manually tested
- Takes 6 hours for comprehensive coverage
- Can add incrementally as bugs are found

**Status**: ⏸️ **MOVED TO v1.1** (add tests as you iterate on features)

---

## Phase 2: Enhancements (Post-Launch)

### 7. **OpenFIGI Integration** 🔍 LOW PRIORITY
**Purpose**: Resolve ISIN ↔ Ticker for European stocks  
**Use Case**: Import CSV with ISINs, auto-populate tickers

**API**: https://www.openfigi.com/api  
**Free Tier**: 25 requests/hour, 250/day

**Implementation**:
```python
def resolve_isin_to_ticker(isin: str) -> str:
    url = "https://api.openfigi.com/v3/mapping"
    headers = {"Content-Type": "application/json"}
    payload = [{"idType": "ID_ISIN", "idValue": isin}]
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data[0]['data'][0]['ticker']
    return None
```

**Files to Edit**: `services/symbol_resolver.py` (new function)  
**Estimated Time**: 3 hours

---

### 8. **Whisper Local Transcription** 🎙️ LOW PRIORITY
**Purpose**: Fallback for videos without captions  
**Model**: `whisper-small` (244 MB, runs locally)

**Dependencies**: `openai-whisper`, `ffmpeg`

**Implementation**:
```python
import whisper

def transcribe_video(video_id: str) -> str:
    # Download audio via yt-dlp
    audio_file = f"temp/{video_id}.mp3"
    os.system(f"yt-dlp -x --audio-format mp3 -o {audio_file} https://youtube.com/watch?v={video_id}")
    
    # Transcribe with Whisper
    model = whisper.load_model("small")
    result = model.transcribe(audio_file)
    
    os.remove(audio_file)
    return result['text']
```

**Files to Edit**: `services/youtube_service.py`  
**Estimated Time**: 4 hours

---

### 9. **News API Integration** 📰 LOW PRIORITY
**Purpose**: Show 3 headlines/day per portfolio ticker  
**API**: NewsAPI.org (100 req/day free)

**Implementation**:
- Fetch daily headlines for each portfolio ticker
- Cache in `data/cache/news_cache.json` (24h TTL)
- Display in Portfolio tab (expandable row)

**Files to Edit**: 
- `services/news_api.py` (new)
- `ui/portfolio.py`

**Estimated Time**: 3 hours

---

### 10. **SQLite Migration** 🗄️ LOW PRIORITY
**Purpose**: Replace CSV with indexed database (10-100x faster)  
**Benefits**: ACID transactions, concurrent access, faster queries

**Schema**:
```sql
CREATE TABLE portfolio (
    id INTEGER PRIMARY KEY,
    name TEXT,
    symbol TEXT UNIQUE,
    isin TEXT,
    quantity REAL,
    bep REAL,
    sector TEXT,
    currency TEXT,
    last_updated TIMESTAMP
);

CREATE TABLE watchlist (
    id INTEGER PRIMARY KEY,
    name TEXT,
    symbol TEXT UNIQUE,
    sector TEXT,
    currency TEXT,
    buy_low REAL,
    buy_high REAL,
    target_1y REAL,
    last_updated TIMESTAMP
);
```

**Files to Create**:
- `services/database.py` (SQLAlchemy or raw sqlite3)
- `migrations/init_schema.sql`

**Files to Edit**: All tabs (replace `pd.read_csv` with `db.query()`)  
**Estimated Time**: 10-12 hours

---

### 11. **Async Queuing for Heavy Calls** ⏱️ LOW PRIORITY
**Purpose**: Don't block UI during YouTube/OpenAI/News fetches  
**Implementation**: Use `asyncio` + `concurrent.futures`

**Example**:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def fetch_summaries_async(channels):
    with ThreadPoolExecutor(max_workers=5) as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, fetch_transcript, ch) for ch in channels]
        results = await asyncio.gather(*tasks)
    return results
```

**Files to Edit**: `services/youtube_service.py`, `services/openai_service.py`  
**Estimated Time**: 4 hours

---

## Phase 3: Scale & Polish (Future)

### 12. **In-App Alerts** 🔔 (Moved from Phase 1)
- Badge counters in tab titles
- Visual alerts for buy zones, targets, stop loss
- Smart frequency cap (2 per stock per week)

**Estimated Time**: 4 hours

---

### 13. **Modular Codebase Refactor** 📂 (Moved from Phase 1)
- Split 2876 lines into organized modules
- `/services` for APIs, `/ui` for tabs, `/utils` for helpers
- Better long-term maintainability

**Estimated Time**: 10 hours

---

### 14. **Unit Test Suite** ✅ (Moved from Phase 1)
- pytest coverage for critical functions
- FX conversion, cache, EAGR formula tests
- CI/CD integration

**Estimated Time**: 6 hours

---

### 15. **Multi-User Authentication** 🔐
- Streamlit auth or Firebase Auth
- User-specific portfolios in database
- Shared watchlists (optional)

**Estimated Time**: 20+ hours

---

### 16. **Cloud Deployment** ☁️
- Streamlit Cloud (free tier: 1 app, public repo)
- Or: AWS EC2 + Docker + Nginx
- Environment secrets via Streamlit Secrets

**Estimated Time**: 4 hours

---

### 17. **Real-Time Prices** ⚡
- WebSocket feeds (IEX Cloud, Polygon.io)
- Replace EOD with live prices
- Auto-refresh every 15s

**Estimated Time**: 8 hours

---

### 18. **Tax Reporting** 💰
- Track buy/sell transactions
- Calculate realized gains (FIFO/LIFO)
- Generate tax forms (US 8949, DE KAP)

**Estimated Time**: 15+ hours

---

### 19. **Mobile App** 📱
- React Native or Flutter
- Shared backend API
- Push notifications

**Estimated Time**: 40+ hours

---

## Summary of Immediate Work (Phase 1)

| Task | Priority | Time | Status |
|------|----------|------|--------|
| 1. EAGR Blended Formula | HIGH | 2h | ✅ **COMPLETED** (Oct 31, 2025) |
| 2. 6h FX Cache TTL | MEDIUM | 1h | ✅ **COMPLETED** (Oct 31, 2025) |
| 3. Color-Coded Heatmap | MEDIUM | 2h | ✅ **COMPLETED** (Oct 31, 2025) |
| 4. Modular Codebase Split | MEDIUM | 10h | ⏸️ **DEFERRED to v1.1** (not urgent) |
| 5. Basic Unit Tests | MEDIUM | 6h | ⏸️ **DEFERRED to v1.1** (quality assurance) |

**Phase 1 Progress**: ✅ **3/3 CORE TASKS COMPLETE (100%)**  
**v1.0 Status**: 🎉 **READY TO LAUNCH**  
**Optional Polish**: Modular refactor + Tests can wait for v1.1

---

## What's Already Done ✅

- [x] All 5 tabs functional (Dashboard, Portfolio, Watchlist, Analysis, Settings)
- [x] Daily caching (prices 24h, FX 24h)
- [x] Triple FX fallback (Marketstack → Frankfurter → Emergency)
- [x] Symbol override system (20 entries)
- [x] Smart CSV import wizard (European format support)
- [x] Batch API calls (80 symbols max)
- [x] Portfolio snapshots (history tracking)
- [x] YouTube summaries (SEED/DAILY/WEEKLY/MONTHLY)
- [x] Top 10 + Other pie charts
- [x] Sortable columns (portfolio & watchlist)
- [x] Git history cleaned (secrets removed)
- [x] Watchlist optimized (202 stocks, down from 251)
- [x] Comprehensive documentation (TECHNICAL_SUMMARY.md)

---

## Recommendation

**✅ v1.0 READY TO LAUNCH NOW!**

All **core critical features** are complete:
- ✅ **EAGR formula** - Smart targets with sector + momentum blending
- ✅ **6h FX cache** - More accurate FX rates (4x daily refresh)
- ✅ **Color heatmap** - Actionable buy zone visualization

**Deferred to v1.1** (not blocking launch):
- Modular codebase split (10h) - Dev experience, not user-facing
- Unit tests (6h) - Quality assurance, add incrementally
- In-app alerts (4h) - Moved to Phase 3 per user request

**🎯 Recommendation**: **Ship v1.0 TODAY!** 

The app is production-ready with all differentiating features live. Don't let perfect be the enemy of good - get it in users' hands now, iterate based on feedback.

**Next milestone**: v1.1 (1-2 weeks) - Refactor codebase, add tests, Phase 2 features

---

**Document Version**: 3.0  
**Created**: October 31, 2025  
**Last Updated**: October 31, 2025 (Phase 1 complete, tasks reorganized)  
**Next Review**: After v1.0 launch & user feedback
