"""
Manufacturer Analytics Dashboard
=================================
End-to-end standalone Streamlit app.
Run: streamlit run manufacturer_dashboard.py
"""

import patch_transformers
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import time
import json
import toml
import math
import logging
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── optional heavy imports (graceful degradation) ───────────────────────────
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from langdetect import detect as lang_detect
    from deep_translator import GoogleTranslator
    TRANS_AVAILABLE = True
except ImportError:
    TRANS_AVAILABLE = False

logging.basicConfig(level=logging.WARNING)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturer Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (dark glassmorphism, Inter font) ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Animated gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 40%, #111827 70%, #0a0a0f 100%);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Sidebar & scrollbar ── */
section[data-testid="stSidebar"] { background: rgba(15,15,25,0.95); }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #007AFF55; border-radius: 4px; }

/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 24px;
    margin-bottom: 18px;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,122,255,0.18);
}

/* ── KPI metric card ── */
.kpi-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px 22px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: default;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,122,255,0.22);
    border-color: rgba(0,122,255,0.4);
}
.kpi-label { color: #8E8E93; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { color: #FFFFFF; font-size: 2rem; font-weight: 700; line-height: 1; }
.kpi-sub   { color: #8E8E93; font-size: 0.78rem; margin-top: 4px; }

/* ── Step badge ── */
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #007AFF, #5856D6);
    color: white; font-size: 0.7rem; font-weight: 700;
    border-radius: 50%; width: 26px; height: 26px;
    margin-right: 8px;
}

/* ── Section heading ── */
.section-heading {
    font-size: 1.35rem; font-weight: 700; color: #F5F5F7;
    margin-bottom: 6px; display: flex; align-items: center;
}
.section-sub { color: #8E8E93; font-size: 0.88rem; margin-bottom: 20px; }

/* ── Insight pill ── */
.pill {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 2px 3px;
}
.pill-pos { background: rgba(52,199,89,0.2);  color: #34C759; border: 1px solid rgba(52,199,89,0.3); }
.pill-neg { background: rgba(255,59,48,0.2);  color: #FF3B30; border: 1px solid rgba(255,59,48,0.3); }
.pill-neu { background: rgba(142,142,147,0.2); color: #8E8E93; border: 1px solid rgba(142,142,147,0.3); }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, rgba(0,122,255,0.15) 0%, rgba(88,86,214,0.12) 50%, rgba(0,122,255,0.08) 100%);
    border: 1px solid rgba(0,122,255,0.25);
    border-radius: 24px;
    padding: 40px 48px;
    margin-bottom: 32px;
    text-align: center;
}
.hero h1 { color: #FFFFFF; font-size: 2.4rem; font-weight: 800; margin: 0; }
.hero p  { color: #8E8E93; font-size: 1.05rem; margin: 10px 0 0; }

/* ── Streamlit overrides ── */
h1,h2,h3 { color: #F5F5F7 !important; }
.stButton > button {
    background: linear-gradient(135deg, #007AFF, #5856D6);
    color: white; border: none; border-radius: 12px;
    padding: 10px 24px; font-weight: 600;
    transition: transform 0.18s, box-shadow 0.18s;
}
.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 8px 24px rgba(0,122,255,0.35);
}
.stSelectbox label, .stMultiSelect label, .stFileUploader label { color: #8E8E93 !important; font-size: 0.85rem !important; }
.stDataFrame { background: rgba(255,255,255,0.03) !important; border-radius: 12px !important; }
div[data-testid="metric-container"] { background: transparent !important; }
.stProgress > div > div { background: linear-gradient(90deg, #007AFF, #5856D6) !important; }
.stInfo    { background: rgba(0,122,255,0.12) !important; color: #7AB7FF !important; border: 1px solid rgba(0,122,255,0.25) !important; border-radius: 10px !important; }
.stSuccess { background: rgba(52,199,89,0.12) !important; color: #34C759 !important; border: 1px solid rgba(52,199,89,0.25) !important; border-radius: 10px !important; }
.stWarning { background: rgba(255,159,10,0.12) !important; color: #FF9F0A !important; border: 1px solid rgba(255,159,10,0.25) !important; border-radius: 10px !important; }
.stError   { background: rgba(255,59,48,0.12)  !important; color: #FF6B60 !important; border: 1px solid rgba(255,59,48,0.25)  !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTS & COLOURS
# ────────────────────────────────────────────────────────────────────────────
COLORS = {
    "positive": "#34C759",
    "negative": "#FF3B30",
    "neutral":  "#8E8E93",
    "primary":  "#007AFF",
    "purple":   "#5856D6",
    "orange":   "#FF9F0A",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#C7C7CC"),
    legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
)
GRID_STYLE = dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)")

# Unified aspect extraction (shared across project)
from aspect_extraction import (
    split_into_sentences, detect_aspect as _shared_detect_aspect,
    detect_all_aspects, run_aspect_sentiment, map_sentiment_label,
    ASPECT_KEYWORDS,
)

FILLER_WORDS = {"lol","ok","k","plz","xd","ha","haha","hmm"}

HF_MODEL_NAME = "unknownexplosion/SentimentABSA-v3"

# ────────────────────────────────────────────────────────────────────────────
# SECRETS
# ────────────────────────────────────────────────────────────────────────────
def load_secrets():
    g_key = os.getenv("GOOGLE_API_KEY", "")
    try:
        s = toml.load(".streamlit/secrets.toml")
        g_key = g_key or s.get("GOOGLE_API_KEY") or s.get("general", {}).get("GOOGLE_API_KEY", "")
    except Exception:
        pass
    return g_key

GOOGLE_API_KEY = load_secrets()

# ────────────────────────────────────────────────────────────────────────────
# NLP HELPERS
# ────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    # Collapse repeated punctuation BEFORE stripping symbols
    # so "!!!" → "!" (preserving the sentence boundary marker)
    text = re.sub(r'([!?.:])\1+', r'\1', text)
    # Remove emojis/symbols but KEEP ! and ? (needed for sentence splitting)
    text = re.sub(r'[^\w\s,.!?]', '', text)
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_meaningless(text: str) -> bool:
    if not text or not isinstance(text, str) or not text.strip():
        return True
    words = set(text.lower().split())
    if words.issubset(FILLER_WORDS):
        return True
    alpha = sum(c.isalpha() for c in text)
    if alpha < 3:
        return True
    alnum = sum(c.isalnum() for c in text)
    if len(text) > 0 and alnum / len(text) < 0.3:
        return True
    return False

def translate_text(text: str) -> str:
    """Translate non-English text to English. Returns original on failure."""
    if not TRANS_AVAILABLE or not text:
        return text
    try:
        lang = lang_detect(text)
        if lang not in ("en", "unknown"):
            return GoogleTranslator(source="auto", target="en").translate(text) or text
    except Exception:
        pass
    return text

def detect_aspect(text: str) -> str:
    """Aspect detector — delegates to the shared aspect_extraction module."""
    return _shared_detect_aspect(text)

def map_label(raw_label: str) -> str:
    """
    Normalize raw model labels. 
    Crucial: Avoid mapping 'LABEL_1' (Neutral) or 'LABEL_2' (Positive) to Negative 
    just because they contain the characters '1' or '2'.
    """
    u = str(raw_label).upper().strip()
    
    # Exact matches for speed and accuracy
    if u in ["POSITIVE", "POS", "LABEL_2", "5", "4", "5 STARS", "4 STARS"]:
        return "Positive"
    if u in ["NEGATIVE", "NEG", "LABEL_0", "1", "2", "1 STAR", "2 STARS"]:
        return "Negative"
    if u in ["NEUTRAL", "NEU", "LABEL_1", "3", "3 STARS"]:
        return "Neutral"
        
    # Substring fallbacks
    if "POSITIVE" in u or "POS" in u: return "Positive"
    if "NEGATIVE" in u or "NEG" in u: return "Negative"
    
    return "Neutral"

def split_into_clauses(text: str):
    """Clause splitter — delegates to the shared spaCy-based splitter."""
    return split_into_sentences(text)

@st.cache_resource(show_spinner=False)
def load_classifier():
    if not TRANSFORMERS_AVAILABLE:
        return None
    return hf_pipeline(
        "sentiment-analysis",
        model=HF_MODEL_NAME,
        device=-1,
        model_kwargs={"low_cpu_mem_usage": False},
    )

def run_sentiment(texts: list, classifier) -> list:
    """Batch sentiment inference. Falls back to Neutral if model missing."""
    if classifier is None:
        return [("Neutral", 0.5)] * len(texts)
    results = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            res = classifier(batch, truncation=True, max_length=512)
            for r in res:
                results.append((map_label(r["label"]), round(r["score"], 4)))
        except Exception:
            results.extend([("Neutral", 0.5)] * len(batch))
    return results

# ────────────────────────────────────────────────────────────────────────────
# PIPELINE — end-to-end on a DataFrame
# ────────────────────────────────────────────────────────────────────────────
def run_full_pipeline(df_raw: pd.DataFrame, model_col: str, review_col: str, date_col: str | None, progress_bar, status_text):
    """
    Runs: clean → translate → sentiment → ABSA.
    Returns (review_df, absa_df).
    """
    df = df_raw[[model_col, review_col] + ([date_col] if date_col else [])].copy()
    df.columns = ["model", "original_review"] + (["date"] if date_col else [])
    df["model"] = df["model"].astype(str).str.strip()

    # ── Step 1: Clean ──
    status_text.markdown("**Step 1/4** — Cleaning text…")
    progress_bar.progress(10)
    df["cleaned"] = df["original_review"].apply(clean_text)
    mask = df["cleaned"].apply(is_meaningless)
    df.loc[mask, "cleaned"] = np.nan

    # ── Step 2: Translate ──
    status_text.markdown("**Step 2/4** — Detecting & translating non-English reviews…")
    progress_bar.progress(25)
    if TRANS_AVAILABLE:
        df["final"] = df["cleaned"].apply(lambda x: translate_text(x) if pd.notna(x) else np.nan)
        df["final"] = df["final"].apply(lambda x: clean_text(x) if pd.notna(x) else np.nan)
    else:
        df["final"] = df["cleaned"]

    # Drop true duplicates per model
    df["_norm"] = df["final"].astype(str).str.lower().str.strip()
    dup = df.duplicated(subset=["model", "_norm"], keep="first")
    df.loc[dup, "final"] = np.nan
    df.drop(columns=["_norm"], inplace=True)

    # ── Step 3: ABSA ──
    status_text.markdown("**Step 3/4** — Extracting aspects (ABSA)…")
    progress_bar.progress(45)
    classifier = load_classifier()
    valid_mask = df["final"].notna() & (df["final"] != "")
    valid_texts = df.loc[valid_mask, "final"].tolist()

    sentiments = [run_aspect_sentiment(t, "General", classifier) for t in valid_texts]
    df.loc[valid_mask, "sentiment_label"]      = [s[0] for s in sentiments]
    df.loc[valid_mask, "sentiment_confidence"] = [s[1] for s in sentiments]

    # ── Step 4: Sentiment ──
    status_text.markdown("**Step 4/4** — Running sentiment model (aspect-conditioned)…")
    progress_bar.progress(75)

    absa_rows = []
    for _, row in df[valid_mask].iterrows():
        text    = row["final"]
        model   = row["model"]
        clauses = split_into_clauses(text)
        if not clauses:
            clauses = [text]

        for clause in clauses:
            # FIX: Use detect_all_aspects to handle multiple aspects in one clause
            aspects = detect_all_aspects(clause)
            if not aspects:
                aspects = ["General"]
                
            for aspect in aspects:
                s_label, s_conf = run_aspect_sentiment(clause, aspect, classifier)
                absa_rows.append({
                    "model":      model,
                    "text":       clause,
                    "aspect":     aspect,
                    "label":      s_label,
                    "confidence": s_conf,
                })

    # Synthetic weekly buckets if no date supplied
    if date_col and "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["week"] = df["date"].dt.to_period("W").astype(str)
        except Exception:
            df["week"] = _synthetic_weeks(df)
    else:
        df["week"] = _synthetic_weeks(df)

    progress_bar.progress(100)
    status_text.markdown("✅ **Pipeline complete!**")

    absa_df = pd.DataFrame(absa_rows)
    return df, absa_df

def _synthetic_weeks(df: pd.DataFrame, reviews_per_week: int = 25) -> pd.Series:
    """Divide rows into synthetic week buckets."""
    n = len(df)
    weeks = []
    for i in range(n):
        week_num = (i // reviews_per_week) + 1
        weeks.append(f"Week {week_num:02d}")
    return pd.Series(weeks, index=df.index)

# ────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ────────────────────────────────────────────────────────────────────────────
def make_line_chart_weekly(df: pd.DataFrame, model_filter: str | None = None):
    """Positive / Neutral / Negative % per week — line chart."""
    df_f = df if model_filter is None else df[df["model"] == model_filter]
    df_f = df_f[df_f["sentiment_label"].notna()]

    grouped = []
    for week, g in df_f.groupby("week"):
        total = len(g)
        pos = (g["sentiment_label"] == "Positive").sum()
        neg = (g["sentiment_label"] == "Negative").sum()
        neu = (g["sentiment_label"] == "Neutral").sum()
        avg_conf = g["sentiment_confidence"].mean() if "sentiment_confidence" in g else 0.5
        grouped.append({
            "Week":     week,
            "Positive": round(pos / total * 100, 1),
            "Negative": round(neg / total * 100, 1),
            "Neutral":  round(neu / total * 100, 1),
            "Avg Confidence": round(avg_conf * 100, 1),
            "Count":    total,
        })
    wdf = pd.DataFrame(grouped).sort_values("Week")

    fig = go.Figure()
    _FILL = {
        COLORS["positive"]: "rgba(52,199,89,0.09)",
        COLORS["negative"]: "rgba(255,59,48,0.09)",
        COLORS["neutral"]:  "rgba(142,142,147,0.09)",
    }
    for s, color in [("Positive", COLORS["positive"]), ("Neutral", COLORS["neutral"]), ("Negative", COLORS["negative"])]:
        fig.add_trace(go.Scatter(
            x=wdf["Week"], y=wdf[s], name=s,
            mode="lines+markers",
            line=dict(color=color, width=2.5, shape="spline"),
            marker=dict(size=6, color=color),
            fill="tozeroy",
            fillcolor=_FILL.get(color, "rgba(142,142,147,0.09)"),
            hovertemplate=f"<b>%{{x}}</b><br>{s}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=wdf["Week"], y=wdf["Count"], name="# Reviews",
        mode="lines", line=dict(color=COLORS["primary"], width=1.5, dash="dot"),
        yaxis="y2", hovertemplate="<b>%{x}</b><br>Reviews: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(title="Sentiment %", range=[0, 105], **GRID_STYLE),
        yaxis2=dict(title="# Reviews", overlaying="y", side="right", showgrid=False, range=[0, wdf["Count"].max() * 3]),
        legend=LEGEND_STYLE,
        margin=dict(t=20, b=40, l=50, r=50),
    )
    return fig

def make_aspect_line_chart(absa_df: pd.DataFrame, model: str):
    """Positive % per aspect per week — multi-line."""
    df_m = absa_df[absa_df["model"] == model]
    if df_m.empty:
        return None

    # We need week info — join from review df stored in session_state
    # Approximate: row index bucketed into weeks
    df_m = df_m.copy().reset_index(drop=True)
    df_m["week"] = [f"W{(i // 20) + 1:02d}" for i in range(len(df_m))]

    top_aspects = df_m["aspect"].value_counts().head(6).index.tolist()
    df_m = df_m[df_m["aspect"].isin(top_aspects)]

    fig = go.Figure()
    palette = [COLORS["primary"], COLORS["positive"], COLORS["orange"], COLORS["purple"], "#FF6CAB", "#5AC8FA"]

    for i, asp in enumerate(top_aspects):
        asp_df = df_m[df_m["aspect"] == asp]
        wdf = asp_df.groupby("week").apply(
            lambda g: round((g["label"] == "Positive").sum() / len(g) * 100, 1),
            include_groups=False
        ).reset_index(name="pos_pct")

        fig.add_trace(go.Scatter(
            x=wdf["week"], y=wdf["pos_pct"],
            name=asp,
            mode="lines+markers",
            line=dict(color=palette[i % len(palette)], width=2, shape="spline"),
            marker=dict(size=5),
            hovertemplate=f"<b>%{{x}}</b><br>{asp} Positive: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        xaxis=dict(**GRID_STYLE),
        yaxis=dict(title="Positive Sentiment %", range=[0, 105], **GRID_STYLE),
        legend=LEGEND_STYLE,
        margin=dict(t=20, b=40, l=50, r=50),
    )
    return fig

def make_radar_chart(absa_df: pd.DataFrame, model: str):
    """Radar chart of positive % per aspect."""
    df_m = absa_df[absa_df["model"] == model]
    if df_m.empty:
        return None

    agg = df_m.groupby("aspect").apply(
        lambda g: round((g["label"] == "Positive").sum() / len(g) * 100, 1),
        include_groups=False
    ).reset_index(name="pos_pct").sort_values("aspect")

    fig = go.Figure(go.Scatterpolar(
        r=agg["pos_pct"].tolist() + [agg["pos_pct"].iloc[0]],
        theta=agg["aspect"].tolist() + [agg["aspect"].iloc[0]],
        fill="toself",
        fillcolor="rgba(0,122,255,0.15)",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(color=COLORS["primary"], size=6),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(255,255,255,0.04)",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#8E8E93", size=9), gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(tickfont=dict(color="#C7C7CC", size=10), gridcolor="rgba(255,255,255,0.1)"),
        ),
        height=320,
        margin=dict(t=20, b=20, l=40, r=40),
        font=dict(family="Inter", color="#C7C7CC"),
    )
    return fig

def make_confidence_histogram(df: pd.DataFrame, model: str):
    df_m = df[(df["model"] == model) & df["sentiment_confidence"].notna()]
    if df_m.empty:
        return None
    fig = px.histogram(
        df_m, x="sentiment_confidence", color="sentiment_label",
        nbins=30,
        color_discrete_map={"Positive": COLORS["positive"], "Negative": COLORS["negative"], "Neutral": COLORS["neutral"]},
        barmode="overlay",
        opacity=0.75,
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=260,
        xaxis_title="Confidence Score",
        yaxis_title="Review Count",
        margin=dict(t=10, b=40),
        legend=LEGEND_STYLE,
    )
    return fig

# ────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ────────────────────────────────────────────────────────────────────────────
def render_kpi_cards(df: pd.DataFrame, absa_df: pd.DataFrame, model: str):
    mdf = df[df["model"] == model]
    valid = mdf[mdf["sentiment_label"].notna()]
    total  = len(mdf)
    pos_p  = round((valid["sentiment_label"] == "Positive").mean() * 100, 1) if len(valid) else 0
    neg_p  = round((valid["sentiment_label"] == "Negative").mean() * 100, 1) if len(valid) else 0
    conf   = round(valid["sentiment_confidence"].mean() * 100, 1) if "sentiment_confidence" in valid and len(valid) else 0

    # Top aspects (Improved logic: use ratios to avoid count bias)
    m_absa = absa_df[absa_df["model"] == model]
    top_pos_asp = top_neg_asp = "—"
    
    if not m_absa.empty:
        # 1. Filter out 'General' and 'Other' for specific insights
        filtered_absa = m_absa[~m_absa["aspect"].isin(["General", "Other"])]
        
        if not filtered_absa.empty:
            # Group by aspect and calculate label percentages
            asp_stats = filtered_absa.groupby("aspect")["label"].value_counts(normalize=True).unstack(fill_value=0)
            asp_counts = filtered_absa["aspect"].value_counts()
            
            # Top Strength: Highest POSITIVE ratio among aspects with >= 3 mentions
            if "Positive" in asp_stats.columns:
                pos_candidates = asp_stats[asp_counts >= 3]["Positive"]
                if not pos_candidates.empty:
                    top_pos_asp = pos_candidates.idxmax()
            
            # Top Issue: Highest NEGATIVE ratio among aspects with >= 2 mentions
            if "Negative" in asp_stats.columns:
                neg_candidates = asp_stats[asp_counts >= 2]["Negative"]
                if not neg_candidates.empty:
                    # We also want to make sure it's actually an "issue" (e.g., > 10% negative)
                    # or at least the most problematic one.
                    top_neg_asp = neg_candidates.idxmax()
                    
                    # Safety check: if the "Top Issue" is 90% positive, it's not really an issue.
                    # In that case, show the one with highest raw negative count if ratio is too low.
                    if neg_candidates.max() < 0.05: # less than 5% negative
                         raw_neg = filtered_absa[filtered_absa["label"] == "Negative"]["aspect"].value_counts()
                         if not raw_neg.empty:
                             top_neg_asp = raw_neg.idxmax()
        else:
            # Fallback to General if no specific aspects found
            top_pos_asp = "General"
            top_neg_asp = "General"

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "📦 Total Reviews", f"{total:,}", "valid after cleaning"),
        (c2, "😊 Positive",      f"{pos_p}%",  "of analysed reviews"),
        (c3, "😤 Negative",      f"{neg_p}%",  "of analysed reviews"),
        (c4, "⭐ Top Strength",  top_pos_asp,  "most praised aspect"),
        (c5, "⚠️ Top Issue",     top_neg_asp,  "most criticised aspect"),
    ]
    for col, label, value, sub in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# AI FEEDBACK (Gemini)
# ────────────────────────────────────────────────────────────────────────────
def generate_ai_feedback(model_name: str, absa_df: pd.DataFrame):
    if not GOOGLE_API_KEY:
        st.error("🔑 Google API key not found in `.streamlit/secrets.toml`.")
        return None
    try:
        from genai_bi import BISummarizer
    except ImportError as e:
        st.error(f"Could not import `genai_bi.py`: {e}")
        return None

    df_model = absa_df[absa_df["model"] == model_name]
    if df_model.empty:
        st.warning("No ABSA records found for this model.")
        return None

    bot = BISummarizer()
    with st.status("🤖 Generating AI Feedback…", expanded=True) as status:
        st.write("Analysing sentiment patterns…")
        result = bot.generate_for_model(model_name, df_model)
        if result:
            status.update(label="✅ Report Generated!", state="complete")
            return result
        else:
            status.update(label="❌ Generation Failed", state="error")
            return None

def render_ai_feedback(summary_json: dict):
    bs = summary_json.get("business_summary", {})

    st.markdown("""
    <div class="glass-card">
        <div class="section-heading">📝 Executive Overview</div>
    """, unsafe_allow_html=True)
    st.info(bs.get("executive_overview", "No overview available."))
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ✅ Key Strengths")
        for item in bs.get("key_strengths", []):
            with st.expander(f"**{item.get('aspect', 'Feature')}**", expanded=True):
                st.markdown(item.get("summary", ""))
                s = item.get("supporting_sentiment", {})
                st.caption(f"Positive: {s.get('positive_share','N/A')} · Negative: {s.get('negative_share','N/A')}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Key Issues")
        for item in bs.get("key_issues", []):
            priority = item.get("priority", "MEDIUM")
            color    = "red" if priority == "HIGH" else "orange" if priority == "MEDIUM" else "blue"
            with st.expander(f"**{item.get('aspect', 'Feature')}** :{color}[{priority}]", expanded=True):
                st.markdown(item.get("summary", ""))
                s = item.get("supporting_sentiment", {})
                st.caption(f"Negative: {s.get('negative_share','N/A')} · Positive: {s.get('positive_share','N/A')}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### 🚀 Actionable Recommendations")
    for rec in bs.get("recommendations", []):
        st.markdown(f"""
        <div style="background:rgba(0,122,255,0.08);padding:16px 20px;border-radius:14px;
                    border-left:4px solid {COLORS['primary']};margin-bottom:10px;">
            <h4 style="margin:0;color:#F5F5F7;">{rec.get('title','Recommendation')}</h4>
            <p style="margin:6px 0 4px;color:#C7C7CC;">{rec.get('description','')}</p>
            <p style="font-size:0.82rem;color:{COLORS['primary']};margin:0;"><b>Expected Impact:</b> {rec.get('expected_impact','')}</p>
        </div>""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TEXTUAL FEEDBACK (rule-based fallback, always available)
# ────────────────────────────────────────────────────────────────────────────
def generate_textual_feedback(df: pd.DataFrame, absa_df: pd.DataFrame, model: str) -> dict:
    mdf = df[df["model"] == model]
    valid = mdf[mdf["sentiment_label"].notna()]
    if valid.empty:
        return {}

    pos_p = round((valid["sentiment_label"] == "Positive").mean() * 100, 1)
    neg_p = round((valid["sentiment_label"] == "Negative").mean() * 100, 1)
    neu_p = round((valid["sentiment_label"] == "Neutral").mean() * 100, 1)

    m_absa = absa_df[absa_df["model"] == model]
    strengths, issues = [], []

    if not m_absa.empty:
        # Dynamic threshold: adapts to the product's overall sentiment profile
        overall_pos_ratio = pos_p / 100.0
        strength_threshold = max(0.35, min(overall_pos_ratio + 0.10, 0.60))
        issue_threshold = max(0.25, min((neg_p / 100.0) + 0.10, 0.40))

        aspect_pos_rates = []
        for asp in m_absa["aspect"].unique():
            asp_df = m_absa[m_absa["aspect"] == asp]
            if len(asp_df) < 3:
                continue
            pos = (asp_df["label"] == "Positive").mean()
            neg = (asp_df["label"] == "Negative").mean()
            aspect_pos_rates.append((asp, pos, neg))
            if pos >= strength_threshold:
                strengths.append((asp, round(pos * 100)))
            if neg >= issue_threshold:
                issues.append((asp, round(neg * 100)))

        # Fallback: if no strengths found, pick top 3 aspects by positive rate
        if not strengths and aspect_pos_rates:
            sorted_by_pos = sorted(aspect_pos_rates, key=lambda x: -x[1])
            for asp, pos, _ in sorted_by_pos[:3]:
                if pos >= 0.30:
                    strengths.append((asp, round(pos * 100)))

    strengths.sort(key=lambda x: -x[1])
    issues.sort(key=lambda x: -x[1])

    tone = "overwhelmingly positive" if pos_p > 75 else "generally positive" if pos_p > 50 else "mixed" if pos_p > 30 else "largely negative"

    return {
        "tone": tone,
        "pos_p": pos_p,
        "neg_p": neg_p,
        "neu_p": neu_p,
        "total": len(valid),
        "strengths": strengths[:5],
        "issues": issues[:5],
    }

def render_textual_feedback(fb: dict, model: str):
    if not fb:
        st.warning("Not enough data to generate feedback.")
        return

    st.markdown(f"""
    <div class="glass-card">
        <div class="section-heading">📋 Automated Analysis — {model}</div>
        <div class="section-sub">Rule-based feedback • Instant • No AI credits needed</div>
        <p style="color:#C7C7CC;line-height:1.7;">
            Based on <b style="color:#F5F5F7">{fb['total']:,}</b> analysed reviews, overall sentiment for
            <b style="color:#F5F5F7">{model}</b> is <b style="color:{
                COLORS['positive'] if 'positive' in fb['tone'] else (COLORS['negative'] if 'negative' in fb['tone'] else COLORS['neutral'])
            }">{fb['tone']}</b>.
            Breakdown: <span class="pill pill-pos">Positive {fb['pos_p']}%</span>
                        <span class="pill pill-neu">Neutral {fb['neu_p']}%</span>
                        <span class="pill pill-neg">Negative {fb['neg_p']}%</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ✅ Top Strengths")
        if fb["strengths"]:
            for asp, pct in fb["strengths"]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
                    <span style="color:#C7C7CC">{asp}</span>
                    <span style="color:{COLORS['positive']};font-weight:700">{pct}% positive</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.caption("No dominant strengths detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Areas to Improve")
        if fb["issues"]:
            for asp, pct in fb["issues"]:
                severity = "HIGH" if pct >= 60 else "MEDIUM" if pct >= 40 else "LOW"
                s_color  = COLORS["negative"] if severity == "HIGH" else COLORS["orange"] if severity == "MEDIUM" else COLORS["neutral"]
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
                    <span style="color:#C7C7CC">{asp}</span>
                    <span style="color:{s_color};font-weight:700">[{severity}] {pct}% negative</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.caption("No major issues detected.")
        st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ────────────────────────────────────────────────────────────────────────────
def main():
    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>📊 Manufacturer Analytics Hub</h1>
        <p>Upload product reviews → run the AI pipeline → get deep insights & weekly trends</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state init ────────────────────────────────────────────────────
    for key in ("review_df", "absa_df", "pipeline_done", "ai_feedback_cache"):
        if key not in st.session_state:
            st.session_state[key]  = None if key != "pipeline_done" else False
            if key == "ai_feedback_cache":
                st.session_state[key] = {}

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Upload & Configure
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-heading">
        <span class="step-badge">1</span> Upload Review Data
    </div>
    <div class="section-sub">CSV must contain at least a model/product column and a review/text column. Date column is optional but enables real weekly trends.</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your CSV here",
        type=["csv"],
        help="Required columns: product/model name + review text. Optional: date.",
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"**Loaded `{uploaded.name}`** — {len(raw_df):,} rows · {len(raw_df.columns)} columns")
        st.dataframe(raw_df.head(5), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        cols = raw_df.columns.tolist()

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Map your columns**")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            model_col = st.selectbox("🏷️ Product / Model column", cols, key="model_col")
        with cc2:
            review_col = st.selectbox("💬 Review / Text column",  cols, key="review_col")
        with cc3:
            date_options = ["(none)"] + cols
            date_sel = st.selectbox("📅 Date column (optional)", date_options, key="date_col")
            date_col = None if date_sel == "(none)" else date_sel
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Run Pipeline button ──────────────────────────────────────────────
        if st.button("🚀 Run Full Analysis", width="stretch"):
            st.session_state["pipeline_done"]    = False
            st.session_state["ai_feedback_cache"] = {}

            progress_bar  = st.progress(0)
            status_text   = st.empty()

            with st.spinner(""):
                review_df, absa_df = run_full_pipeline(
                    raw_df, model_col, review_col, date_col,
                    progress_bar, status_text,
                )

            st.session_state["review_df"]     = review_df
            st.session_state["absa_df"]       = absa_df
            st.session_state["pipeline_done"] = True
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # SECTIONS 2–4: Results (only if pipeline ran)
    # ════════════════════════════════════════════════════════════════════════
    if not st.session_state["pipeline_done"] or st.session_state["review_df"] is None:
        if not uploaded:
            st.markdown("""
            <div style="text-align:center;padding:60px 0;color:#8E8E93;">
                <div style="font-size:3rem;margin-bottom:12px;">📂</div>
                <div style="font-size:1.1rem;">Upload a CSV above to get started</div>
                <div style="font-size:0.85rem;margin-top:6px;">Supports any product reviews — not just Apple</div>
            </div>
            """, unsafe_allow_html=True)
        return

    review_df = st.session_state["review_df"]
    absa_df   = st.session_state["absa_df"]
    models    = sorted(review_df["model"].unique().tolist())

    st.success(f"✅ Pipeline complete — {len(review_df):,} reviews · {len(absa_df):,} aspect clauses · {len(models)} product(s)")

    # ── Model selector (tabs if ≤6 models, dropdown otherwise) ──────────────
    st.markdown("---")
    if len(models) <= 6:
        tabs = st.tabs([f"**{m}**" for m in models])
        model_tab_pairs = list(zip(models, tabs))
    else:
        selected_model = st.selectbox("Select Product Model", models)
        model_tab_pairs = [(selected_model, st.container())]

    for model, container in model_tab_pairs:
        with container:
            _render_model_section(model, review_df, absa_df)


def _render_model_section(model, review_df, absa_df):
    """Renders KPI + charts + feedback for one model."""

    # ────── KPI Cards ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="section-heading" style="margin-top:20px">
        <span class="step-badge">2</span> Key Metrics — {model}
    </div>
    """, unsafe_allow_html=True)
    render_kpi_cards(review_df, absa_df, model)

    # ────── Weekly Line Charts ─────────────────────────────────────────────
    st.markdown("""
    <div class="section-heading" style="margin-top:28px">
        <span class="step-badge">3</span> Weekly Sentiment Trends
    </div>
    <div class="section-sub">Line charts showing how sentiment evolves over time (or review batches if no date column was provided).</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Overall Sentiment Over Time (Positive / Neutral / Negative %)")
    fig_line = make_line_chart_weekly(review_df, model)
    st.plotly_chart(fig_line, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Aspect Positive % Over Time")
        fig_asp = make_aspect_line_chart(absa_df, model)
        if fig_asp:
            st.plotly_chart(fig_asp, width="stretch")
        else:
            st.caption("Not enough aspect data.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Aspect Satisfaction Radar")
        fig_radar = make_radar_chart(absa_df, model)
        if fig_radar:
            st.plotly_chart(fig_radar, width="stretch")
        else:
            st.caption("Not enough data.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Confidence histogram
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Model Confidence Distribution")
    fig_hist = make_confidence_histogram(review_df, model)
    if fig_hist:
        st.plotly_chart(fig_hist, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    # ────── Textual Feedback ────────────────────────────────────────────────
    st.markdown("""
    <div class="section-heading" style="margin-top:28px">
        <span class="step-badge">4</span> Automated Textual Feedback
    </div>
    """, unsafe_allow_html=True)

    fb = generate_textual_feedback(review_df, absa_df, model)
    render_textual_feedback(fb, model)

    # ────── AI Feedback (Gemini) ────────────────────────────────────────────
    st.markdown("""
    <div class="section-heading" style="margin-top:28px">
        <span class="step-badge">5</span> AI-Generated Executive Report
    </div>
    <div class="section-sub">Powered by Gemini 2.5 Flash — generates executive overview, strengths, issues, and recommendations.</div>
    """, unsafe_allow_html=True)

    cache_key = f"ai_{model}"
    # Show cached result if available
    if cache_key in st.session_state.get("ai_feedback_cache", {}):
        render_ai_feedback(st.session_state["ai_feedback_cache"][cache_key])
    else:
        if st.button(f"✨ Generate AI Report for {model}", key=f"ai_btn_{model}"):
            result = generate_ai_feedback(model, absa_df)
            if result:
                st.session_state["ai_feedback_cache"][cache_key] = result
                st.rerun()

    # ────── Download ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⬇️ Download Results")
    dc1, dc2 = st.columns(2)
    with dc1:
        m_reviews = review_df[review_df["model"] == model]
        st.download_button(
            "📥 Review-Level Sentiment CSV",
            data=m_reviews.to_csv(index=False).encode("utf-8"),
            file_name=f"{model.replace(' ','_')}_sentiment.csv",
            mime="text/csv",
            width="stretch",
        )
    with dc2:
        m_absa = absa_df[absa_df["model"] == model]
        st.download_button(
            "📥 Aspect-Level (ABSA) CSV",
            data=m_absa.to_csv(index=False).encode("utf-8"),
            file_name=f"{model.replace(' ','_')}_absa.csv",
            mime="text/csv",
            width="stretch",
        )


if __name__ == "__main__":
    main()
