# Technical Architecture Reference

> Deep-dive supplement to the main README. For setup and usage, see [README.md](README.md).

---

## Data Flow Diagram

```
User Input
    │
    ├── CSV Upload (Manufacturer Hub / Playground)
    │       │
    │       └─► run_full_pipeline() / _mfg_run_pipeline()
    │
    ├── Reddit Scrape (Reddit Model Scout / Auto-Scraper)
    │       │
    │       └─► RedditScraper.run()  →  3-col DataFrame
    │                                        │
    │                                        └─►_mfg_run_pipeline()
    │
    └── Pre-built dataset (Live Dashboard / Manufacturer Report)
            │
            └─► sentiment_pipeline.py (offline, run once)
                        │
                        └─► outputs/*.csv  (loaded by app.py)


Pipeline Steps (shared across all sources):
─────────────────────────────────────────────────────────────────

  clean_text()         →  remove URLs, HTML, non-word chars,
                           control chars, repeated punctuation

  is_meaningless()     →  drop filler words, < 3 alpha chars,
                           < 30% alphanumeric ratio

  translate_text()     →  langdetect → GoogleTranslator (if non-EN)

  deduplication        →  drop exact-match reviews per model
                           (case-insensitive normalised key)

  DeBERTa inference    →  batch 32, truncate to 512 tokens
                           map raw label → Positive / Negative / Neutral

  split_into_clauses() →  re.split on [.!?] and conjunctions
                           (but / however / though)

  detect_aspect()      →  keyword-count over ASPECT_KEYWORDS dict
                           returns highest-scoring aspect

  weekly bucketing     →  real dates via pd.Period("W")
                           OR synthetic "Week NN" (25 reviews/bucket)
```

---

## Module Dependency Graph

```
app.py
 ├── sentiment_pipeline.py  (outputs/ CSVs pre-generated)
 ├── reddit_scraper.py      (RedditScraper, run_pipeline_on_new_data)
 └── genai_bi.py            (BISummarizer → MongoDB)

manufacturer_dashboard.py
 └── genai_bi.py            (BISummarizer)

weekly_scheduler.py
 ├── reddit_scraper.py      (RedditScraper)
 └── sentiment_pipeline.py  (pipeline functions)
```

---

## Session State Keys

### `app.py` — Manufacturer Hub (Page 6)

| Key | Type | Purpose |
|---|---|---|
| `mfg_review_df` | `pd.DataFrame \| None` | Review-level sentiment output |
| `mfg_absa_df` | `pd.DataFrame \| None` | ABSA clause-level output |
| `mfg_done` | `bool` | Whether pipeline has completed |
| `mfg_ai_cache` | `dict[str, Any]` | Gemini report per model, keyed `mfg_ai_{model}` |

### `app.py` — Reddit Model Scout (Page 7)

| Key | Type | Purpose |
|---|---|---|
| `scout_review_df` | `pd.DataFrame \| None` | Review-level output after pipeline |
| `scout_absa_df` | `pd.DataFrame \| None` | ABSA output after pipeline |
| `scout_done` | `bool` | Whether pipeline has completed |
| `scout_ai_cache` | `dict[str, Any]` | Gemini report per model |
| `scout_query` | `str \| None` | Last search query typed by user |
| `scout_csv` | `bytes \| None` | Raw scraped CSV (for download) |

### `manufacturer_dashboard.py`

| Key | Type | Purpose |
|---|---|---|
| `review_df` | `pd.DataFrame \| None` | Pipeline output |
| `absa_df` | `pd.DataFrame \| None` | ABSA output |
| `pipeline_done` | `bool` | Pipeline completion flag |
| `ai_feedback_cache` | `dict` | Gemini report cache |

---

## CSS Design System

Both UIs share a dark glassmorphism design language.

### Colour Tokens

| Token | Hex | Usage |
|---|---|---|
| `positive` | `#34C759` | Positive sentiment badges, pills, charts |
| `negative` | `#FF3B30` | Negative sentiment, HIGH priority issues |
| `neutral` | `#8E8E93` | Neutral sentiment, subtitles |
| `primary` | `#007AFF` | Buttons, links, primary accents |
| `purple` | `#5856D6` | Step badges (gradient end), AI section |
| `orange` | `#FF9F0A` | MEDIUM priority issues, warnings |

### Key CSS Classes

| Class | Description |
|---|---|
| `.glass-card` | Glassmorphism container (backdrop-filter blur) |
| `.kpi-card` | KPI metric tile with hover lift animation |
| `.step-badge` | Circular gradient number badge |
| `.pill-pos / neg / neu` | Inline sentiment percentage pills |
| `.mfg-hero` | Dark-gradient hero banner |
| `.mfg-glass` | Manufacturer Hub glass panel |

---

## Plotly Chart Conventions

All charts use the shared `PLOTLY_LAYOUT` / `MFG_PLOTLY_LAYOUT` dict:

```python
dict(
    paper_bgcolor="rgba(0,0,0,0)",   # transparent — sits over CSS bg
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#C7C7CC"),
)
```

Grid lines: `rgba(255,255,255,0.06)` (barely visible on dark bg).

Line charts use `shape="spline"` for smooth curves and `fill="tozeroy"` with 9% alpha fill.

---

## Reddit Scraper State File

`outputs/scraped/scraper_state.json`

```json
{
  "seen_ids": ["abc123", "def456", "..."],   // last 10,000 post IDs
  "last_run": "2026-04-07T20:00:00+00:00",
  "total_scraped": 4821,
  "run_count": 12
}
```

The scout page (`render_reddit_scout`) does **not** read this file — it scrapes a fresh set for any arbitrary user query, with dedup only within that single run (using an in-memory `seen_ids` set).

---

## MongoDB Schema

### Database: `sentiment_analysis_db`
### Collection: `manufacturer_bi_summaries`

```json
{
  "_id": ObjectId("..."),
  "model": "iPhone 15 Pro",
  "business_summary": {
    "executive_overview": "...",
    "key_strengths": [
      {
        "aspect": "Camera",
        "summary": "Users consistently praise...",
        "supporting_sentiment": {
          "positive_share": "78%",
          "negative_share": "8%"
        }
      }
    ],
    "key_issues": [
      {
        "aspect": "Battery",
        "priority": "HIGH",
        "summary": "Many users report...",
        "supporting_sentiment": {
          "negative_share": "55%",
          "positive_share": "20%"
        }
      }
    ],
    "recommendations": [
      {
        "title": "Optimise Background Process Management",
        "description": "...",
        "linked_aspects": ["Battery"],
        "expected_impact": "20-30% reduction in negative battery sentiment"
      }
    ]
  }
}
```

Documents are **upserted** by `model` — re-generating a report overwrites the previous one.

---

## Aspect Keyword Map (Full)

Used by both `detect_aspect()` helpers in `app.py` and `manufacturer_dashboard.py`:

```python
{
  "Camera":         ["camera","photo","picture","image quality","selfie","lens","sensor","night mode","clarity"],
  "Battery":        ["battery","battery life","charge","charging","drains","power","sot"],
  "Performance":    ["performance","speed","lag","slow","fast","smooth","chip","gpu","cpu","processor","freeze","hang"],
  "Display":        ["display","screen","oled","brightness","resolution","refresh rate","retina","120hz"],
  "Design & Build": ["design","build","material","durability","sleek","thin","lightweight","scratch","aesthetics"],
  "Software & OS":  ["ios","macos","software","update","bug","crash","glitch","ui","ux","icloud"],
  "Audio":          ["audio","sound","speaker","bass","microphone","mic","call quality"],
  "Connectivity":   ["wifi","bluetooth","network","5g","signal","connectivity","hotspot"],
  "Storage":        ["storage","memory","ram","128gb","256gb","512gb","1tb"],
  "Price":          ["price","cost","expensive","overpriced","cheap","value","worth"],
  "Heating":        ["heat","heating","hot","overheats","thermal"],
}
```

Scoring: each keyword that appears in the clause text adds 1 point to the corresponding aspect. The aspect with the highest score wins. Ties default to whichever appears first in the dict. Clauses with no keyword match are labelled `"Other"`.

---

## Environment Variables

| Variable | Purpose | Alternative |
|---|---|---|
| `REDDIT_CLIENT_ID` | Reddit API client ID | `[reddit].client_id` in `secrets.toml` |
| `REDDIT_CLIENT_SECRET` | Reddit API client secret | `[reddit].client_secret` in `secrets.toml` |
| `REDDIT_USER_AGENT` | Reddit user agent string | `[reddit].user_agent` in `secrets.toml` |
| `GOOGLE_API_KEY` | Gemini 2.5 Flash API key | `GOOGLE_API_KEY` in `secrets.toml` |
| `MONGO_URI` | MongoDB Atlas connection string | `[general].MONGO_URI` in `secrets.toml` |

Environment variables take precedence over `secrets.toml` values.
