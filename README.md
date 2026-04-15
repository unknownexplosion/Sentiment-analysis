<div align="center">

# 🎯 Sentiment Analysis Platform
### End-to-End ABSA · Reddit Intelligence · Manufacturer Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-DeBERTa_v3-orange?style=for-the-badge)](https://huggingface.co/unknownexplosion/SentimentAnalysisog)
[![Gemini](https://img.shields.io/badge/AI-Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Decode public opinion at aspect level · Live-scrape Reddit · Generate executive AI reports · Track weekly trends**

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Architecture](#-system-architecture)
4. [Pages & Modules](#-pages--modules)
5. [File Structure](#-file-structure)
6. [Quick Start](#-quick-start)
7. [Configuration](#-configuration)
8. [Pipeline Deep Dive](#-pipeline-deep-dive)
9. [Model Information](#-model-information)
10. [API & Integration](#-api--integration)
11. [Outputs](#-outputs)
12. [Troubleshooting](#-troubleshooting)

---

## 🌟 Overview

This platform is a **production-grade NLP pipeline** built on top of Microsoft DeBERTa v3. It goes far beyond simple positive/negative classification — it performs **Aspect-Based Sentiment Analysis (ABSA)**, attributing sentiment to specific product features like Camera, Battery, Performance, and Price.

The platform has two complementary entry points:

| Entry Point | Use Case |
|---|---|
| `streamlit run app.py` | Full research dashboard (Apple-focused + general review analysis) |
| `streamlit run manufacturer_dashboard.py` | Standalone manufacturer analytics hub (upload any CSV) |

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **ABSA Engine** | Splits reviews into clauses and labels each clause's sentiment per detected aspect |
| **Multi-language support** | Auto-detects and translates non-English reviews to English before analysis |
| **Reddit Model Scout** | Search **any product** → scrape Reddit live → run the full AI pipeline in one click |
| **Manufacturer Analytics Hub** | Upload any product-review CSV → KPI cards → weekly line graphs → AI report |
| **Weekly Auto-Scraper** | Scheduled Reddit scraping (every Sunday) with dedup and state management |
| **Gemini AI Reports** | Executive overview, key strengths/issues, and actionable recommendations via Gemini 2.5 Flash |
| **Business Intelligence** | MongoDB-backed BI summaries per model, accessible across sessions |
| **Live Dashboard** | Interactive Plotly charts for the pre-built Apple dataset |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │ CSV Upload   │   │ Reddit PRAW  │   │ final_dataset.csv│    │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘    │
└─────────┼──────────────────┼────────────────────┼──────────────┘
          │                  │                    │
          ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PROCESSING PIPELINE                         │
│                                                                 │
│  ① Text Cleaning  →  ② Language Detection & Translation         │
│         ↓                                                       │
│  ③ Deduplication  →  ④ DeBERTa v3 Sentiment Inference          │
│         ↓                                                       │
│  ⑤ Clause Splitting  →  ⑥ ABSA Aspect Detection               │
│         ↓                                                       │
│  ⑦ Weekly Bucketing  →  ⑧ Aggregation & Stats                 │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION                             │
│                                                                 │
│  • KPI Cards          • Weekly Line Charts    • Radar Charts    │
│  • Donut Charts       • Aspect Bar Charts     • Histograms      │
│  • Textual Feedback   • Gemini AI Reports     • CSV Downloads   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                          STORAGE                                │
│                                                                 │
│  outputs/sentiment_output.csv   outputs/absa_training_dataset  │
│  outputs/scraped/               MongoDB Atlas (BI summaries)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📄 Pages & Modules

### `app.py` — Primary Application

The main Streamlit app. Navigate between pages from the left sidebar.

#### Page 1 — Project Overview
Explains the mission, shows the model pipeline diagram (Graphviz), and displays key performance metrics of the fine-tuned DeBERTa model.

#### Page 2 — Live Dashboard
Interactive analytics on the pre-built Apple product dataset:
- Sentiment distribution donut chart
- Rating trend bar chart
- Aspect-level stacked bar chart (ABSA breakdown)
- Per-product filtering via dropdown

#### Page 3 — Model Playground
Test the fine-tuned model interactively:
- **Single review mode**: Paste any text → get overall sentiment + clause-level ABSA breakdown table
- **CSV batch mode**: Upload a CSV, pick the review column, get full ABSA analysis with downloadable results

#### Page 4 — Manufacturer Report
Renders the pre-generated Markdown report from `outputs/manufacturer_recommendations.md`. Run `sentiment_pipeline.py` to regenerate it.

#### Page 5 — Business Intelligence
AI-generated executive summaries powered by Gemini 2.5 Flash, persisted to MongoDB:
- Select a model from the database or ABSA dataset
- Click **Generate Report** → structured JSON with executive overview, strengths, issues, recommendations
- Stored in the `manufacturer_bi_summaries` collection

#### Page 6 — Manufacturer Analytics Hub
Upload any product-review CSV and run the complete end-to-end pipeline:

| Step | Action |
|---|---|
| 1 | Upload CSV → map model column, review column, optional date column |
| 2 | Click **Run Full Analysis** |
| 3 | View per-product KPI cards (total reviews, positive %, negative %, top praised/criticised aspect) |
| 4 | Weekly sentiment line charts (Positive / Neutral / Negative % over time) |
| 5 | Aspect positive % line chart (top 6 aspects over time) |
| 6 | Aspect satisfaction radar chart |
| 7 | Rule-based automated textual feedback |
| 8 | Gemini AI executive report (one click, cached per session) |
| 9 | Download enriched sentiment CSV or ABSA clause CSV |

#### Page 7 — Reddit Model Scout ⭐ New
Search **any product** and scrape Reddit reviews on demand:

| Step | Action |
|---|---|
| 1 | Enter product/model name (e.g. `Samsung Galaxy S24`, `Sony WH-1000XM5`) |
| 2 | Choose time period (week / month / year / all) and max posts per query |
| 3 | Optionally add extra subreddits (e.g. `gadgets, Android`) |
| 4 | Click **Scrape Reddit & Analyse** |
| 5 | Download the raw 3-column CSV (`model_name`, `review`, `date`) |
| 6 | View full analytics: KPI cards, donut, aspect bar, weekly line charts, radar, textual & AI reports |

> **CSV Format**: `model_name | review | date` — this CSV is also forwarded automatically into the Manufacturer Analytics Hub pipeline.

---

### `manufacturer_dashboard.py` — Standalone Dashboard

A fully self-contained standalone Streamlit app with the same upload-and-analyse capability as Page 6 above, but with its own dark-glassmorphism UI and no dependency on the pre-built Apple dataset.

```bash
streamlit run manufacturer_dashboard.py
```

---

### `reddit_scraper.py` — Reddit Data Collection

Automated Reddit scraper built on the official **PRAW** (Python Reddit API Wrapper) library.

**Features:**
- Scrapes from 8 target subreddits (`r/apple`, `r/iphone`, `r/MacBook`, `r/iPad`, etc.)
- Runs search queries across `r/all` for fresh review posts
- Pulls post body + top-N comments per post
- Deduplicates via post ID across runs (persisted in `scraper_state.json`)
- Maps posts to product models via keyword matching
- Outputs: `outputs/scraped/reddit_reviews_all.csv` (cumulative) + weekly timestamped file

**Usage:**
```bash
python reddit_scraper.py                # Full run + pipeline
python reddit_scraper.py --no-pipeline  # Scrape only, skip pipeline
python reddit_scraper.py --dry-run      # Test credentials only
```

---

### `weekly_scheduler.py` — Automated Scheduling

Runs the Reddit scraper every Sunday at 2:00 AM automatically.

```bash
python weekly_scheduler.py            # Start infinite loop (keeps running)
python weekly_scheduler.py --now      # Trigger job once immediately
python weekly_scheduler.py --status   # Show last/next run and post counts
```

---

### `sentiment_pipeline.py` — Offline Pipeline

Batch pipeline for processing the full Apple review dataset from a CSV file.

**Pipeline steps:**
1. `load_data()` — read CSV, keep model + review columns
2. `preprocess_reviews()` — clean, remove meaningless text
3. `translate_and_clean()` — detect language, translate to English
4. `handle_duplicates()` — deduplicate per model
5. `analyze_sentiment()` — run DeBERTa inference in batches of 32
6. `generate_absa_dataset()` — extract aspect-level sentiment using spaCy sentences
7. `aggregate_model_stats()` — compute per-model Positive/Negative/Neutral % + top keywords
8. `generate_feedback_report()` — structured feedback with recommendations
9. `plot_results()` — save PNG charts to `outputs/plots/`

```bash
python sentiment_pipeline.py
```

---

### `genai_bi.py` — Gemini AI Business Intelligence

Provides the `BISummarizer` class that calls Gemini 2.5 Flash to generate structured executive reports.

**Output JSON structure:**
```json
{
  "model": "iPhone 15 Pro",
  "business_summary": {
    "executive_overview": "...",
    "key_strengths": [{"aspect": "Camera", "summary": "...", "supporting_sentiment": {...}}],
    "key_issues":   [{"aspect": "Battery", "priority": "HIGH", "summary": "..."}],
    "recommendations": [{"title": "...", "description": "...", "expected_impact": "..."}]
  }
}
```

Supports retries with exponential back-off (15s → 30s → 60s → 120s) on rate-limit errors.

---

## 📁 File Structure

```
Sentiment-analysis/
│
├── app.py                        # Main Streamlit app (7 pages)
├── manufacturer_dashboard.py     # Standalone Manufacturer Hub
│
├── sentiment_pipeline.py         # Offline batch NLP pipeline
├── reddit_scraper.py             # Reddit PRAW scraper
├── weekly_scheduler.py           # Sunday auto-scheduler
├── genai_bi.py                   # Gemini AI BI report generator
│
├── train_absa_model.py           # DeBERTa fine-tuning script
├── run_full_system.py            # End-to-end runner (pipeline + training)
├── ingest_vectors.py             # Vector store ingestion helper
│
├── final_dataset.csv             # Pre-built Apple review dataset (~40k rows)
│
├── outputs/
│   ├── sentiment_output.csv      # Review-level sentiment with labels
│   ├── absa_training_dataset.csv # Clause-level ABSA dataset
│   ├── per_model_summary.csv     # Aggregated stats per model
│   ├── feedback_report.csv       # Structured feedback per model
│   ├── manufacturer_recommendations.md  # Human-readable report
│   ├── plots/                    # PNG charts (global sentiment, per-model, keywords)
│   └── scraped/
│       ├── reddit_reviews_all.csv        # Cumulative Reddit data
│       ├── reddit_YYYY-WNN.csv           # Weekly snapshots
│       ├── scraper_state.json            # Seen post IDs + last run time
│       └── schedule.json                 # Scheduler state
│
├── assets/
│   └── apple_logo.png
│
├── .streamlit/
│   └── secrets.toml              # 🔑 API keys (see Configuration)
│
├── requirements.txt
├── DEPLOYMENT.md
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/unknownexplosion/Sentiment-analysis.git
cd Sentiment-analysis

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
pip install praw toml               # For Reddit scraper

# Download spaCy model (required for offline pipeline)
python -m spacy download en_core_web_sm
```

### 2. Configure Secrets

Create `.streamlit/secrets.toml`:

```toml
# ── Reddit API (required for Reddit Model Scout + auto-scraper) ──────────────
[reddit]
client_id     = "your_client_id"
client_secret = "your_client_secret"
user_agent    = "SentimentScoutBot/1.0 by YourUsername"

# ── Google Gemini (required for AI Executive Reports) ────────────────────────
GOOGLE_API_KEY = "your_gemini_api_key"

# ── MongoDB Atlas (required for Business Intelligence page) ──────────────────
[general]
MONGO_URI = "mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/"
```

### 3. Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The app works even without secrets configured — optional features (Reddit scraping, AI reports, MongoDB BI) will show setup guidance rather than errors.

---

### Optional: Run the Offline Pipeline

```bash
python sentiment_pipeline.py
```

This processes `final_dataset.csv` and writes outputs to `outputs/`. The **Live Dashboard** and **Manufacturer Report** pages read from these files.

---

### Optional: Start the Weekly Scheduler

```bash
# Runs continuously, triggers every Sunday at 2 AM
python weekly_scheduler.py

# Or trigger once right now
python weekly_scheduler.py --now
```

---

## ⚙️ Configuration

### Reddit API Credentials

1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Click **Create App** → choose **script**
3. Set redirect URI to `http://localhost:8080`
4. Copy `client_id` (under the app name) and `client_secret`
5. Paste into `.streamlit/secrets.toml` under `[reddit]`

### Google Gemini API Key

1. Visit [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → Create API key in a new project
3. Paste the key as `GOOGLE_API_KEY` in `secrets.toml`

### MongoDB Atlas (for BI page)

1. Create a free cluster at [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Click **Connect → Drivers → Python** → copy the connection string
3. Replace `<password>` with your user password
4. Paste as `MONGO_URI` in `secrets.toml` under `[general]`

> The database `sentiment_analysis_db` and collection `manufacturer_bi_summaries` are created automatically on first use.

---

## 🔬 Pipeline Deep Dive

### Text Cleaning

Every review passes through this sequence:

```
Raw text
  → Strip URLs (http/https)
  → Strip HTML tags (<...>)
  → Remove non-word characters (emojis, symbols)
  → Collapse whitespace
  → Reduce repeated punctuation (!! → !)
  → Meaningless check (filler words, < 3 alpha chars, < 30% alphanumeric ratio)
```

### Language Translation

```python
lang = langdetect.detect(text)
if lang != "en":
    text = GoogleTranslator(source="auto", target="en").translate(text)
```

Supported: all languages covered by Google Translate (~100+). Falls back silently on failure.

### Sentiment Inference

The fine-tuned DeBERTa model (`unknownexplosion/SentimentAnalysisog` on Hugging Face) outputs raw labels that are mapped:

| Raw Label | Display Label |
|---|---|
| Contains `5`, `4`, `POS`, `POSITIVE` | **Positive** |
| Contains `1`, `2`, `NEG`, `NEGATIVE` | **Negative** |
| Everything else | **Neutral** |

Inference runs in batches of 32 with `truncation=True, max_length=512`.

### Aspect Detection

For each clause (split on `.`, `!`, `?`, `but`, `however`, `though`), the system checks which keyword set matches most:

| Aspect | Sample Keywords |
|---|---|
| **Camera** | camera, photo, selfie, lens, sensor, night mode |
| **Battery** | battery, charge, drains, power, sot |
| **Performance** | lag, slow, fast, smooth, chip, gpu, freeze |
| **Display** | screen, oled, brightness, refresh rate, 120hz |
| **Design & Build** | design, durability, sleek, scratch, lightweight |
| **Software & OS** | ios, macos, update, bug, crash, glitch |
| **Audio** | speaker, bass, mic, call quality |
| **Connectivity** | wifi, bluetooth, 5g, signal |
| **Storage** | storage, ram, 128gb, 256gb |
| **Price** | price, expensive, overpriced, value |
| **Heating** | heat, overheats, thermal |

### Weekly Bucketing

If a `date` column is provided, reviews are grouped by calendar week (`pd.Period("W")`). Without a date column, reviews are divided into synthetic "Week 01", "Week 02", ... buckets (25 reviews per week). This allows weekly line charts to render regardless of data source.

---

## 🤖 Model Information

| Property | Value |
|---|---|
| **Base Model** | Microsoft DeBERTa-v3-small |
| **Hosted on** | [huggingface.co/unknownexplosion/SentimentAnalysisog](https://huggingface.co/unknownexplosion/SentimentAnalysisog) |
| **Task** | Multi-class Sentiment Classification (Positive / Negative / Neutral) |
| **Training Data** | Apple product reviews (Amazon, Reddit, App Store) |
| **Training Device** | NVIDIA T4 GPU · 3 Epochs |
| **Accuracy** | **91.5%** |
| **F1-Score** | **0.915** |
| **Precision** | **91.6%** |
| **Max Token Length** | 512 |
| **Inference Device** | CPU (device=-1), compatible with MPS/CUDA |

**Why DeBERTa over BERT?**
DeBERTa uses *disentangled attention* — it separately encodes word content and word position. This gives it superior understanding of negations (`"not good"` ≠ `"good"`) and dependent phrases that are critical for review sentiment.

**Baseline comparison:**

| Model | Accuracy |
|---|---|
| VADER (rule-based) | ~65% |
| BERT base | ~78% |
| **DeBERTa v3 (fine-tuned)** | **91.5%** |

---

## 🔌 API & Integration

### Using the Model Directly

```python
from transformers import pipeline

clf = pipeline(
    "sentiment-analysis",
    model="unknownexplosion/SentimentAnalysisog",
    device=-1,  # CPU; set to 0 for CUDA
)

text = "The camera is amazing but the battery drains too fast."
result = clf(text)
# [{'label': 'POSITIVE', 'score': 0.93}]
```

### Using the Sentiment Pipeline Programmatically

```python
from sentiment_pipeline import (
    preprocess_reviews,
    translate_and_clean,
    handle_duplicates,
    analyze_sentiment,
    generate_absa_dataset,
    aggregate_model_stats,
)
import pandas as pd

df = pd.read_csv("my_reviews.csv")
df.columns = ["model", "original_review"]

df = preprocess_reviews(df)
df = translate_and_clean(df)
df = handle_duplicates(df)
df, clf = analyze_sentiment(df)
absa_df = generate_absa_dataset(df, clf)
stats   = aggregate_model_stats(df)
```

### Generating AI Reports Programmatically

```python
from genai_bi import BISummarizer
import pandas as pd

absa_df = pd.read_csv("outputs/absa_training_dataset.csv")
bot     = BISummarizer()

records = absa_df[absa_df["model_name"] == "iPhone 15 Pro"].to_dict("records")
report  = bot.generate_for_model("iPhone 15 Pro", records[:80])
bot.save_to_mongodb(report)
```

---

## 📊 Outputs

| File | Description |
|---|---|
| `outputs/sentiment_output.csv` | One row per review: `model`, `original_review`, `final_review`, `sentiment_label`, `sentiment_score` |
| `outputs/absa_training_dataset.csv` | One row per aspect-clause: `model_name`, `text`, `aspect`, `label` |
| `outputs/per_model_summary.csv` | Per-model aggregated stats: positive/negative/neutral %, top keywords |
| `outputs/feedback_report.csv` | Structured feedback per model: summary, strengths, weaknesses, recommendations |
| `outputs/manufacturer_recommendations.md` | Human-readable Markdown report for all models |
| `outputs/plots/*.png` | Global sentiment distribution, per-model counts, top keywords |
| `outputs/scraped/reddit_reviews_all.csv` | Cumulative Reddit-scraped posts + comments |

---

## 🛠 Troubleshooting

### App crashes on import / model not loading

```bash
pip install --upgrade transformers torch
```

If on Apple Silicon Mac with MPS issues:
```bash
# Force CPU in app.py / manufacturer_dashboard.py
device = -1  # already set — ensure no override
```

---

### Reddit credentials error

```
ValueError: Reddit API credentials not found.
```

Ensure `.streamlit/secrets.toml` has:
```toml
[reddit]
client_id     = "..."
client_secret = "..."
user_agent    = "..."
```

And restart the Streamlit app after editing.

---

### MongoDB DNS / cluster does not exist

```
🔌 Cannot reach MongoDB Atlas — the cluster DNS record does not exist.
```

1. Log in to [MongoDB Atlas](https://www.mongodb.com/atlas) and verify your cluster is **active**
2. Get a fresh connection string (Connect → Drivers → Python)
3. Paste it into the text field shown on the Business Intelligence page — it saves automatically

---

### Gemini quota errors (429)

The `BISummarizer` class handles this automatically with exponential back-off (15s → 30s → 60s → 120s). If errors persist:
- Check your [Google AI Studio quota](https://aistudio.google.com/app/u/0/plan_information)
- Consider upgrading to a paid tier for higher rate limits

---

### Translation not working

```bash
pip install langdetect deep-translator
```

If `langdetect` raises `LangDetectException: No features in text` on short texts, this is expected — the pipeline falls back to the original text silently.

---

### spaCy model not found

```bash
python -m spacy download en_core_web_sm
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and commit: `git commit -m 'Add my feature'`
4. Push and open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using **DeBERTa** · **Streamlit** · **Plotly** · **Gemini** · **PRAW**

</div>
