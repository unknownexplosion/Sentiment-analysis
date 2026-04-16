import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from collections import Counter
# from fpdf import FPDF # GenAI removed

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Transformers import failed: {e}")

try:
    from langdetect import detect as lang_detect
    from deep_translator import GoogleTranslator
    MFG_TRANS_AVAILABLE = True
except ImportError:
    MFG_TRANS_AVAILABLE = False

try:
    import spacy as _spacy
    _spacy_nlp = _spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

import re
import time
import pymongo
import certifi
import toml
from retraining_dashboard import render_retraining_center

# --- Helper Functions for ABSA ---
def _map_label_to_display(label: str):
    """Normalize raw model label into Positive / Negative / Neutral + color."""
    label_upper = label.upper()
    color = COLORS['neutral']
    display_label = "Neutral"

    if any(x in label_upper for x in ["5", "4", "POS", "POSITIVE"]):
        display_label = "Positive"
        color = COLORS['positive']
    elif any(x in label_upper for x in ["1", "2", "NEG", "NEGATIVE"]):
        display_label = "Negative"
        color = COLORS['negative']

    return display_label, color

def _split_into_clauses(text: str):
    """Naive split into clauses for more granular sentiment."""
    # Break on '.', '!', '?', 'but', 'however', 'though'
    parts = re.split(r'(?i)\bbut\b|\bhowever\b|\bthough\b|[.!?]', text)
    clauses = [p.strip() for p in parts if p.strip()]
    return clauses

def _detect_aspect(text: str) -> str:
    """Heuristic aspect detector for Apple reviews."""
    t = text.lower()

    aspect_keywords = {
    "Camera": [
        "camera", "cameras", "photo", "photos", "picture", "pictures",
        "image quality", "picture quality", "clarity", "sharpness",
        "selfie", "front camera", "rear camera", "telephoto", "ultrawide",
        "portrait mode", "macro mode", "night mode", "hdr", "stabilization",
        "optical zoom", "digital zoom", "lens", "sensor", "exposure"
    ],

    "Battery": [
        "battery", "battery life", "battery backup", "charge", "charging",
        "charging speed", "fast charging", "wireless charging", "charger",
        "power adapter", "power consumption",
        "drains fast", "drains quickly", "loses charge", "dies quickly",
        "needs frequent charging", "screen on time", "sot"
    ],

    "Performance": [
        "performance", "speed", "lag", "slow", "fast", "smooth", "snappy",
        "responsive", "responsiveness", "multitasking", "freeze", "freezes",
        "stutter", "stutters", "hang", "hangs", "choppy",
        "processor", "chip", "gpu", "cpu",
        "a14", "a15", "a16", "a17", "m1", "m2", "m3", "m3 pro", "m3 max",
        "overheats", "heats up", "thermal throttle", "thermal throttling"
    ],

    "Display": [
        "display", "screen", "lcd", "oled", "super retina", "retina",
        "brightness", "contrast", "color accuracy", "colour accuracy",
        "resolution", "refresh rate", "120hz", "90hz", "60hz", "promotion",
        "vivid colors", "washed out", "sunlight visibility", "glare",
        "viewing angles", "pixel density"
    ],

    "Design & Build": [
        "design", "build", "build quality", "material", "aluminium", "metal",
        "durability", "durable", "sleek", "thin", "lightweight", "premium feel",
        "matte finish", "glossy finish", "scratch", "scratches easily",
        "look", "looks", "feel in hand", "aesthetics"
    ],

    "Software & OS": [
        "ios", "macos", "software", "system", "os", "update", "updates",
        "bug", "bugs", "crash", "crashes", "glitch", "glitches",
        "freezes", "freeze", "ui", "ux", "user interface", "notifications",
        "apple ecosystem", "continuity", "handoff", "airdrop", "icloud"
    ],

    "Audio": [
        "audio", "sound", "speaker", "speakers", "bass", "treble",
        "loudness", "microphone", "mic", "call quality", "voice clarity",
        "stereo speakers", "muffled audio", "tinny sound"
    ],

    "Connectivity": [
        "wifi", "wi-fi", "bluetooth", "network", "cellular", "5g", "lte",
        "signal", "connectivity", "hotspot", "airdrop disconnect",
        "network drops", "weak signal", "unstable wifi"
    ],

    "Storage": [
        "storage", "space", "memory", "ram",
        "32gb", "64gb", "128gb", "256gb", "512gb", "1tb",
        "running out of space", "not enough storage"
    ],

    "Price": [
        "price", "pricing", "cost", "expensive", "overpriced", "too costly",
        "cheap", "value for money", "worth the price", "not worth it",
        "premium pricing"
    ],

    "Heating / Thermals": [
        "heat", "heating", "heats", "heats up", "gets hot", "overheats",
        "thermal throttling", "hot while charging", "hot during gaming",
    ],

    "Other": []
}

    best_aspect = "Other"
    best_score = 0

    for aspect, kws in aspect_keywords.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best_score = score
            best_aspect = aspect

    return best_aspect


# --- App Config ---
st.set_page_config(
    page_title="Apple Sentiment Analysis",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Minimalist Apple-like Design + Manufacturer Hub dark theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .main {
        background-color: #FBFBFD;
    }
    h1, h2, h3 {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1D1D1F;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #007AFF;
        color: white;
        border-radius: 18px;
        border: none;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.02);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stMetricLabel { color: #86868B; }
    .stMetricValue { color: #1D1D1F; }

    /* ── Manufacturer Hub styles ── */
    .mfg-hero {
        background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0d1117 100%);
        border: 1px solid rgba(0,122,255,0.25);
        border-radius: 24px;
        padding: 36px 44px;
        margin-bottom: 28px;
        text-align: center;
    }
    .mfg-hero h1 { color: #FFFFFF !important; font-size: 2.2rem; font-weight: 800; margin: 0; font-family: 'Inter', sans-serif; }
    .mfg-hero p  { color: #8E8E93; font-size: 1rem; margin: 10px 0 0; }
    .mfg-glass {
        background: rgba(13,17,23,0.7);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        padding: 22px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .mfg-glass:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 32px rgba(0,122,255,0.15);
    }
    .mfg-kpi {
        background: linear-gradient(145deg, rgba(13,17,23,0.9), rgba(17,24,39,0.8));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .mfg-kpi:hover { transform: translateY(-3px); box-shadow: 0 12px 32px rgba(0,122,255,0.2); border-color: rgba(0,122,255,0.35); }
    .mfg-kpi-label { color: #8E8E93; font-size: 0.75rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px; }
    .mfg-kpi-value { color: #FFFFFF; font-size: 1.9rem; font-weight: 700; line-height: 1; }
    .mfg-kpi-sub   { color: #636366; font-size: 0.75rem; margin-top: 4px; }
    .mfg-step-badge {
        display: inline-flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #007AFF, #5856D6);
        color: white; font-size: 0.68rem; font-weight: 700;
        border-radius: 50%; width: 24px; height: 24px; margin-right: 8px;
    }
    .mfg-sh { font-size: 1.25rem; font-weight: 700; color: #F5F5F7 !important; margin-bottom: 4px; display: flex; align-items: center; }
    .mfg-sub { color: #8E8E93; font-size: 0.85rem; margin-bottom: 18px; }
    .mfg-pill-pos { display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;margin:2px 3px;background:rgba(52,199,89,0.2);color:#34C759;border:1px solid rgba(52,199,89,0.3); }
    .mfg-pill-neg { display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;margin:2px 3px;background:rgba(255,59,48,0.2);color:#FF3B30;border:1px solid rgba(255,59,48,0.3); }
    .mfg-pill-neu { display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;margin:2px 3px;background:rgba(142,142,147,0.2);color:#8E8E93;border:1px solid rgba(142,142,147,0.3); }
    </style>
""", unsafe_allow_html=True)

# Apple Color Palette
COLORS = {
    'positive': '#34C759',  # Apple Green
    'negative': '#FF3B30',  # Apple Red
    'neutral': '#8E8E93',   # Apple Grey
    'primary': '#007AFF',   # Apple Blue
    'background': '#FFFFFF'
}

# --- Manufacturer Hub Constants ---
MFG_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#C7C7CC"),
)
MFG_GRID   = dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)")
MFG_LEGEND = dict(bgcolor="rgba(255,255,255,0.04)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
                  orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

MFG_ASPECT_KEYWORDS = {
    "Camera":         ["camera","photo","picture","image quality","selfie","lens","sensor","night mode","clarity"],
    "Battery":        ["battery","battery life","charge","charging","drains","power","sot"],
    "Performance":    ["performance","speed","lag","slow","fast","smooth","chip","gpu","cpu","processor","freeze","hang"],
    "Display":        ["display","screen","oled","brightness","resolution","refresh rate","retina","120hz"],
    "Design & Build": ["design","build","material","durability","sleek","thin","lightweight","scratch"],
    "Software & OS":  ["ios","macos","software","update","bug","crash","glitch","ui","ux","icloud"],
    "Audio":          ["audio","sound","speaker","bass","microphone","mic","call quality"],
    "Connectivity":   ["wifi","bluetooth","network","5g","signal","connectivity","hotspot"],
    "Storage":        ["storage","memory","ram","128gb","256gb","512gb","1tb"],
    "Price":          ["price","cost","expensive","overpriced","cheap","value","worth"],
    "Heating":        ["heat","heating","hot","overheats","thermal"],
}
MFG_FILLER = {"lol","ok","k","plz","xd","ha","haha","hmm"}

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('outputs/sentiment_output.csv')
        absa_df = pd.read_csv('outputs/absa_training_dataset.csv') if os.path.exists('outputs/absa_training_dataset.csv') else pd.DataFrame()
        return df, absa_df
    except FileNotFoundError:
        return None, None

df, absa_df = load_data()

def _render_scraper_sidebar():
    """Shows Reddit scraper status + manual trigger in the sidebar."""
    import json
    from pathlib import Path

    state_file  = Path("outputs/scraped/scraper_state.json")
    scraped_csv = Path("outputs/scraped/reddit_reviews_all.csv")

    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        last_run = state.get("last_run", "Never")[:10] if state.get("last_run") else "Never"
        total    = state.get("total_scraped", 0)
        runs     = state.get("run_count", 0)
        st.caption(f"Last run: `{last_run}`")
        st.caption(f"Posts scraped: `{total:,}` ({runs} runs)")
    else:
        st.caption("Not yet configured")

    # Check if Reddit creds are set
    try:
        secrets = toml.load(".streamlit/secrets.toml")
        r = secrets.get("reddit", {})
        creds_ok = bool(r.get("client_id", "").strip())
    except Exception:
        creds_ok = False

    if creds_ok:
        if st.button("▶ Run Scraper Now", key="sidebar_scrape_btn"):
            with st.spinner("Scraping Reddit…"):
                try:
                    from reddit_scraper import RedditScraper, run_pipeline_on_new_data
                    scraper = RedditScraper()
                    new_df = scraper.run()
                    if new_df is not None:
                        st.success(f"✅ {len(new_df)} new posts scraped!")
                    else:
                        st.info("No new posts since last run.")
                except Exception as e:
                    st.error(f"Scraper error: {e}")
    else:
        st.caption("⚠️ Add Reddit credentials to `secrets.toml` to enable")


# --- Page logic ---
def main():
    if df is None:
        # Manufacturer Hub doesn't need pre-built data — still allow it
        with st.sidebar:
            st.title("Navigation")
            page = st.radio("Go to", ["Manufacturer Analytics Hub"], label_visibility="collapsed")
            st.markdown("---")
            st.caption("v2.3 • Manufacturer Hub added")
        render_manufacturer_hub()
        return

    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to",
            ["Project Overview", "Live Dashboard", "Model Playground",
             "Manufacturer Report", "Business Intelligence",
             "Manufacturer Analytics Hub", "Reddit Model Scout", "Model Retraining Center", "Reddit Data Warehouse"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.info("💡 **Tip:** Use the 'Playground' to test your own text.")
        st.markdown("---")

        # ── Reddit Scraper Status ──
        st.markdown("**🤖 Reddit Auto-Scraper**")
        _render_scraper_sidebar()

        st.markdown("---")
        st.caption("v2.5 • Real-time Retraining Center")


    if page == "Project Overview":
        render_overview()
    elif page == "Live Dashboard":
        render_dashboard(df, absa_df)
    elif page == "Model Playground":
        render_playground()
    elif page == "Manufacturer Report":
        render_report()
    elif page == "Business Intelligence":
        render_bi_dashboard()
    elif page == "Manufacturer Analytics Hub":
        render_manufacturer_hub()
    elif page == "Reddit Model Scout":
        render_reddit_scout()
    elif page == "Model Retraining Center":
        render_retraining_center()
    elif page == "Reddit Data Warehouse":
        render_reddit_data()

# --- Page: Business Intelligence ---
def render_bi_dashboard():
    st.markdown("## 🧠 Business Intelligence Hub")
    st.caption("AI-Generated Executive Summaries (Powered by Gemini 1.5)")

    # 1. Connect to DB
    # Load secrets again just to be safe or use st.secrets logic if consistent
    mongo_uri = st.secrets.get("general", {}).get("MONGO_URI") or os.getenv("MONGO_URI")
    
    if not mongo_uri:
        # Fallback to local config loading if st.secrets not populated yet in dev
        try:
            secrets = toml.load(".streamlit/secrets.toml")
            mongo_uri = secrets.get("MONGO_URI") or secrets.get("general", {}).get("MONGO_URI")
        except:
            pass
            
    if not mongo_uri:
        st.error("🚨 MongoDB URI not found. Please configure .streamlit/secrets.toml")
        return

    try:
        client = pymongo.MongoClient(
            mongo_uri,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=5000,  # Fail fast instead of hanging
        )
        # Force an early connection check so DNS errors surface immediately
        client.admin.command("ping")
        db = client.get_database("sentiment_analysis_db")
        col = db["manufacturer_bi_summaries"]
        
        # 2. Fetch Models
        db_models = col.distinct("model") or []
        
        # Merge with available data from ABSA
        csv_models = []
        if not absa_df.empty:
            if 'model_name' in absa_df.columns:
                csv_models = absa_df['model_name'].dropna().unique().tolist()
            elif 'model' in absa_df.columns:
                csv_models = absa_df['model'].dropna().unique().tolist()
                
        # Union and Sort
        models = sorted(list(set(db_models + csv_models)))
        
        if not models:
            st.warning("No models found in database or dataset.")
            return

        selected_model = st.selectbox("Select Product Model", models)
        
        # 3. Fetch Data
        record = col.find_one({"model": selected_model})
        
        # --- Generation Controls ---
        col_gen1, col_gen2 = st.columns([3, 1])
        with col_gen1:
            if not record:
                st.info("No report exists for this model yet.")
        with col_gen2:
            if st.button("✨ Generate Report", type="primary" if not record else "secondary"):
                try:
                    from genai_bi import BISummarizer
                    
                    # Filter data for this model
                    # Ensure alignment of column names
                    model_col = 'model_name' if 'model_name' in absa_df.columns else 'model'
                    df_model = absa_df[absa_df[model_col] == selected_model]

                    if df_model.empty:
                        st.error("No ABSA data available for this model.")
                    else:
                        with st.status("🤖 AI Agent Generating Report...", expanded=True) as status:
                            bi_bot = BISummarizer()
                            status.write("Analyzing sentiment patterns...")
                            # Generate using fair prompt data (all records)
                            summary_json = bi_bot.generate_for_model(selected_model, df_model)
                            
                            if summary_json:
                                status.write("Saving to database...")
                                bi_bot.save_to_mongodb(summary_json)
                                status.update(label="✅ Report Generated!", state="complete")
                                time.sleep(1)
                                st.rerun()
                            else:
                                status.update(label="❌ Generation Failed", state="error")
                except ImportError:
                    st.error("Could not import genai_bi.py")
                except Exception as e:
                    st.error(f"Generation Error: {e}")

        if not record or "business_summary" not in record:
            return
            
        summary = record["business_summary"]
        
        # --- UI Layout ---
        
        # A. Executive Overview
        st.markdown("### 📝 Executive Overview")
        st.info(summary.get("executive_overview", "No overview available."))
        
        st.divider()
        
        # B. Strengths & Issues
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("✅ Key Strengths")
            strengths = summary.get("key_strengths", [])
            if not strengths:
                st.write("No specific strengths listed.")
            for item in strengths:
                with st.expander(f"**{item.get('aspect', 'Feature')}**", expanded=True):
                    st.write(item.get("summary", ""))
                    stats = item.get("supporting_sentiment", {})
                    st.caption(f"Positive: {stats.get('positive_share', 'N/A')} • Negative: {stats.get('negative_share', 'N/A')}")
        
        with c2:
            st.subheader("⚠️ Key Issues")
            issues = summary.get("key_issues", [])
            if not issues:
                st.write("No major issues detected.")
            for item in issues:
                priority = item.get("priority", "MEDIUM")
                p_color = "red" if priority == "HIGH" else "orange" if priority == "MEDIUM" else "blue"
                
                with st.expander(f"**{item.get('aspect', 'Feature')}** :{p_color}[{priority}]", expanded=True):
                    st.write(item.get("summary", ""))
                    stats = item.get("supporting_sentiment", {})
                    st.caption(f"Negative: {stats.get('negative_share', 'N/A')} • Positive: {stats.get('positive_share', 'N/A')}")

        st.divider()

        # C. Recommendations
        st.subheader("🚀 Actionable Recommendations")
        recs = summary.get("recommendations", [])
        
        for rec in recs:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #007AFF;">
                    <h4 style="margin:0; color: #1D1D1F;">{rec.get('title', 'Recommendation')}</h4>
                    <p style="margin-top: 5px; color: #424245;">{rec.get('description', '')}</p>
                    <p style="font-size: 0.9em; color: #007AFF; margin-bottom: 0;"><b>Expected Impact:</b> {rec.get('expected_impact', '')}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # D. JSON View (Optional)
        with st.expander("🛠️ View Raw JSON Data"):
            st.json(record)

    except Exception as e:
        err_str = str(e)
        # Detect DNS / cluster-not-found errors specifically
        if "DNS" in err_str or "does not exist" in err_str or "NXDOMAIN" in err_str or "ServerSelectionTimeoutError" in err_str:
            st.error(
                "🔌 **Cannot reach MongoDB Atlas** — the cluster DNS record does not exist.\n\n"
                "This usually means the cluster was **deleted** or the connection string is **outdated**."
            )
            st.markdown("### How to fix")
            st.markdown(
                "1. Log in to [MongoDB Atlas](https://www.mongodb.com/atlas) and verify your cluster is **active**.\n"
                "2. Click **Connect → Drivers (Python)** and copy the new connection string.\n"
                "3. Paste it below **or** update `.streamlit/secrets.toml` and restart the app."
            )
            new_uri = st.text_input(
                "Paste your new MongoDB connection string here:",
                placeholder="mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/",
                type="password",
            )
            if new_uri:
                try:
                    import toml as _toml
                    secrets_path = ".streamlit/secrets.toml"
                    try:
                        cfg = _toml.load(secrets_path)
                    except Exception:
                        cfg = {"general": {}}
                    cfg.setdefault("general", {})["MONGO_URI"] = new_uri
                    with open(secrets_path, "w") as f:
                        _toml.dump(cfg, f)
                    st.success("✅ New URI saved to `secrets.toml`. Click **Rerun** (top-right menu) or refresh the page to reconnect.")
                    st.button("🔄 Rerun Now", on_click=st.rerun)
                except Exception as save_err:
                    st.warning(f"Could not auto-save URI: {save_err}. Please update `.streamlit/secrets.toml` manually.")
        else:
            st.error(f"Database Error: {e}")


# --- Page: Project Overview ---
def render_overview():
    # Centered Logo
    col1, col2, col3 = st.columns([1, 0.2, 1])
    with col2:
        st.image("assets/apple_logo.png", width=100)
        
    st.markdown("<div style='text-align: center; padding-bottom: 40px;'><h1>Apple Sentiment Analysis</h1><p style='color: #86868B; font-size: 1.2rem;'>Decoding customer perception with fine-tuned Transformers.</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("The Mission")
        st.markdown("""
        In a world of noise, understanding the **signal** is key. 
        
        This project moves beyond simple star ratings. We use **Aspect-Based Sentiment Analysis (ABSA)** to dissect exactly *what* users love or hate about apple products — be it the **Battery**, **Camera**, or **Price**.
        
        We don't just ask *"Is it good?"*
        We ask *"Why is it good?"*
        """)
        
        # --- Live Metrics Extraction ---
        acc_str = "91.5"
        f1_str = "0.915"
        prec_str = "0.916"
        try:
            import json, os
            metrics_path = "outputs/fine_tuned_absa_model/metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    if "eval_accuracy" in metrics:
                        acc_str = f"{metrics['eval_accuracy'] * 100:.1f}"
                        f1_str = f"{metrics.get('eval_f1', 0.915):.3f}"
                        prec_str = f"{metrics.get('eval_precision', 0.916):.3f}"
        except Exception:
            pass
        # -------------------------------
        
        st.markdown("### 🏆 Model Performance")
        st.markdown(f"""
        Our fine-tuned **DeBERTa v3** model achieves industry-leading metrics:
        
        | Metric | Score |
        | :--- | :--- |
        | **Accuracy** | <span style='color:{COLORS['positive']}'>**{acc_str}%**</span> |
        | **F1-Score** | {f1_str} |
        | **Precision** | {prec_str} |
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("How It Works")
        
        # Native Graphviz Chart (Robust & Clean)
        st.graphviz_chart("""
            digraph {
                rankdir="TB";
                node [shape=box, style="filled,rounded", fillcolor="#ffffff", fontname="sans-serif", penwidth=0];
                edge [color="#8E8E93"];
                bgcolor="transparent";
                
                A [label="Raw Reviews"];
                B [label="Preprocessing"];
                C [label="Aspect Extraction", shape=diamond, fillcolor="#e3f2fd"];
                D [label="DeBERTa Model", fillcolor="#e8f5e9"];
                E [label="Sentiment Score"];
                F [label="Dashboard"];

                A -> B;
                B -> C;
                C -> D [label="Input"];
                D -> E;
                E -> F;
            }
        """)
        
        with st.expander("🛠️ See Tech Stack Details"):
            st.markdown("""
            *   **Core Model:** Microsoft DeBERTa V3 Small (Fine-tuned)
            *   **Embedding:** Contextual Transformer Embeddings
            *   **Frontend:** Streamlit & Plotly
            *   **Preprocessing:** Spacy & Regex
            """)

# --- Page: Dashboard ---
def render_dashboard(df, absa_df):
    st.markdown("## 📊 Live Analytics Dashboard")

    # Filter by Model (Restored Feature)
    if 'model' in df.columns:
        model_list = ['All'] + sorted(df['model'].dropna().unique().tolist())
        selected_model = st.selectbox("Select Model Source", model_list)
        
        if selected_model != 'All':
            df = df[df['model'] == selected_model]
            # Filter ABSA data too if it has model info, otherwise leave it or filter loosely
            if 'model_name' in absa_df.columns:
                 absa_df = absa_df[absa_df['model_name'] == selected_model]
    
    # 1. Top Level Metrics
    total_reviews = len(df)
    avg_rating = df['sentiment_score'].mean()
    pos_pct = (df['sentiment_label'] == 'Positive').mean() * 100
    neg_pct = (df['sentiment_label'] == 'Negative').mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{total_reviews:,}")
    c2.metric("Average Rating", f"{avg_rating:.1f} ★")
    c3.metric("Positive Sentiment", f"{pos_pct:.1f}%", delta_color="normal")
    c4.metric("Negative Sentiment", f"{neg_pct:.1f}%", delta_color="inverse")
    
    st.divider()

    # 2. Charts Row 1
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Sentiment Distribution")
        # Custom Donut Chart
        sentiment_counts = df['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Label', 'Count']
        
        fig_donut = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Label',
            color='Label',
            color_discrete_map={'Positive': COLORS['positive'], 'Negative': COLORS['negative'], 'Neutral': COLORS['neutral']},
            hole=0.6
        )
        fig_donut.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, width="stretch")

    with col2:
        st.subheader("Rating Trends")
        star_counts = df['sentiment_score'].value_counts().sort_index().reset_index()
        star_counts.columns = ['Stars', 'Count']
        
        fig_bar = px.bar(
            star_counts, 
            x='Stars', 
            y='Count',
            text_auto=True,
            color_discrete_sequence=[COLORS['primary']]
        )
        fig_bar.update_layout(xaxis_type='category', margin=dict(t=0, b=0, l=0, r=0), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, width="stretch")

    # 3. ABSA Section
    if not absa_df.empty:
        st.divider()
        st.subheader("💡 Aspect Analysis (What people are talking about)")
        
        # Aggregate sentiment by aspect
        aspect_sentiment = pd.crosstab(absa_df['aspect'], absa_df['label'], normalize='index') * 100
        aspect_sentiment = aspect_sentiment.reset_index()
        
        # Sort by aspect frequency to show most relevant first
        aspect_counts = absa_df['aspect'].value_counts().head(8).index
        aspect_sentiment = aspect_sentiment[aspect_sentiment['aspect'].isin(aspect_counts)]

        fig_absa = go.Figure()
        for label, color in [('Negative', COLORS['negative']), ('Neutral', COLORS['neutral']), ('Positive', COLORS['positive'])]:
            if label in aspect_sentiment.columns:
                fig_absa.add_trace(go.Bar(
                    y=aspect_sentiment['aspect'],
                    x=aspect_sentiment[label],
                    name=label,
                    orientation='h',
                    marker_color=color
                ))

        fig_absa.update_layout(
            barmode='stack', 
            title="Sentiment per Feature (Top 8)",
            xaxis_title="Percentage %",
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_absa, width="stretch")

# --- Page: Playground ---
def render_playground():
    st.markdown(
        "<div style='text-align: center;'><h2>🧠 Model Playground</h2>"
        "<p>Test the fine-tuned DeBERTa model with your own text or upload a CSV for aspect-based analysis.</p></div>",
        unsafe_allow_html=True
    )

    # ---- Load model once for this page ----
    if not TRANSFORMERS_AVAILABLE:
        st.error("⚠️ Transformers library could not be loaded. Please check your installation.")
        return

    try:
        HF_MODEL_NAME = "unknownexplosion/SentimentAnalysisog"
        st.sidebar.info(f"Model: {HF_MODEL_NAME} (Hugging Face)")
        final_model_name = HF_MODEL_NAME

        with st.spinner("Loading sentiment model..."):
            # forcing device=-1 (CPU) avoids "meta tensor" errors on Mac/Accelerate
            classifier = pipeline("sentiment-analysis", model=final_model_name, device=-1, model_kwargs={"low_cpu_mem_usage": False})
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return

    # ===========================================================
    #  OPTION 1 – SINGLE REVIEW (Detailed ABSA-like breakdown)
    # ===========================================================
    st.subheader("🔍 Analyze Single Review (with Aspects)")

    user_input = st.text_area(
        "Enter a review:",
        height=150,
        placeholder="e.g., The camera is amazing but the battery drains too fast when gaming."
    )

    if st.button("Analyze Sentiment", width="stretch"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing review..."):
                try:
                    # ---- Overall sentiment ----
                    overall_result = classifier(user_input)[0]
                    raw_label = overall_result["label"]
                    score = overall_result["score"]
                    display_label, color = _map_label_to_display(raw_label)

                    # ---- Clause-level analysis ----
                    clauses = _split_into_clauses(user_input)
                    clause_rows = []

                    for c in clauses:
                        r = classifier(c)[0]
                        disp_label, _ = _map_label_to_display(r["label"])
                        aspect = _detect_aspect(c)
                        clause_rows.append({
                            "Clause": c,
                            "Aspect": aspect,
                            "Raw Label": r["label"],
                            "Sentiment": disp_label,
                            "Confidence": round(r["score"], 4),
                        })

                    clause_df = pd.DataFrame(clause_rows)

                    # ---- Nice overall card ----
                    st.markdown(f"""
                    <div style="
                        background-color: {color}20;
                        padding: 20px;
                        border-radius: 12px;
                        border: 2px solid {color};
                        text-align: center;
                        margin-top: 20px;">
                        <h3 style="color: {color}; margin:0;">Overall Sentiment: {display_label}</h3>
                        <p style="margin:0; font-weight:bold;">Confidence: {score:.2%}</p>
                        <p style="margin-top:8px; color:#555;">Model raw label: {raw_label}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Detailed clause/aspect table ----
                    st.markdown("### 🔍 Clause & Aspect Breakdown")
                    st.dataframe(clause_df, width="stretch")

                    # ---- Simple manufacturer insight by aspect ----
                    st.markdown("### 🧾 Insight for Manufacturer (by Aspect)")

                    pos_df = clause_df[clause_df["Sentiment"] == "Positive"]
                    neg_df = clause_df[clause_df["Sentiment"] == "Negative"]

                    insights = []

                    if not pos_df.empty:
                        pos_group = pos_df.groupby("Aspect")["Clause"].apply(
                            lambda x: "; ".join(x.tolist())
                        )
                        for aspect, text in pos_group.items():
                            insights.append(f"✅ **{aspect}**: {text}")

                    if not neg_df.empty:
                        neg_group = neg_df.groupby("Aspect")["Clause"].apply(
                            lambda x: "; ".join(x.tolist())
                        )
                        for aspect, text in neg_group.items():
                            insights.append(f"⚠️ **{aspect}**: {text}")

                    if not insights:
                        insights.append("Overall tone is neutral with no strong praise or complaints detected.")

                    for line in insights:
                        st.markdown(line)

                except Exception as e:
                    st.error(f"Error during analysis: {e}")

    st.markdown("---")

    # ===========================================================
    #  OPTION 2 – CSV UPLOAD (Batch Aspect-Based Sentiment)
    # ===========================================================
    st.subheader("📤 Upload CSV for Batch Aspect-Based Sentiment Analysis")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("📄 **Preview of Uploaded File:**")
        st.dataframe(df.head())

        # Let user pick which column has review text
        review_column = st.selectbox(
            "Select the column that contains the review text:",
            df.columns.tolist()
        )

        if st.button("Run Batch ABSA on CSV"):
            with st.spinner("Running model on all reviews (clause + aspect level)..."):
                try:
                    overall_sentiments = []
                    aspect_rows = []

                    for idx, row in df.iterrows():
                        text = str(row[review_column])
                        if not text or not text.strip():
                            overall_sentiments.append({"index": idx, "overall_raw": None,
                                                       "overall_sentiment": None,
                                                       "overall_confidence": None})
                            continue

                        # Overall sentiment per review
                        overall_res = classifier(text)[0]
                        o_disp, _ = _map_label_to_display(overall_res["label"])
                        overall_sentiments.append({
                            "index": idx,
                            "overall_raw": overall_res["label"],
                            "overall_sentiment": o_disp,
                            "overall_confidence": round(overall_res["score"], 4),
                        })

                        # Clause-level ABSA
                        clauses = _split_into_clauses(text)
                        for c in clauses:
                            if not c.strip():
                                continue
                            r = classifier(c)[0]
                            disp_label, _ = _map_label_to_display(r["label"])
                            aspect = _detect_aspect(c)

                            aspect_rows.append({
                                "review_index": idx,
                                "review_text": text,
                                "clause": c,
                                "aspect": aspect,
                                "sentiment": disp_label,
                                "raw_label": r["label"],
                                "confidence": round(r["score"], 4),
                            })

                    # Merge overall sentiment back to df
                    overall_df = pd.DataFrame(overall_sentiments).set_index("index")
                    df["overall_raw"] = df.index.map(overall_df["overall_raw"])
                    df["overall_sentiment"] = df.index.map(overall_df["overall_sentiment"])
                    df["overall_confidence"] = df.index.map(overall_df["overall_confidence"])

                    st.success("Batch aspect-based sentiment analysis completed!")

                    st.markdown("### ✅ Review-level Sentiment (with overall scores)")
                    st.dataframe(df, width="stretch")

                    # Create aspect-level DataFrame
                    aspects_df = pd.DataFrame(aspect_rows)
                    if aspects_df.empty:
                        st.info("No aspects detected in the uploaded text.")
                        return

                    st.markdown("### 🔍 Clause & Aspect-Level Details")
                    st.dataframe(aspects_df.head(100), width="stretch")

                    # ---- Aggregate by aspect & sentiment ----
                    st.markdown("### 📊 Aggregated Sentiment by Aspect")

                    agg = (
                        aspects_df
                        .groupby(["aspect", "sentiment"])
                        .size()
                        .unstack(fill_value=0)
                        .reset_index()
                    )

                    # Make sure all three sentiment columns exist
                    for col in ["Positive", "Neutral", "Negative"]:
                        if col not in agg.columns:
                            agg[col] = 0

                    # Total per aspect for percentage
                    agg["Total"] = agg["Positive"] + agg["Neutral"] + agg["Negative"]
                    for col in ["Positive", "Neutral", "Negative"]:
                        agg[col + "_pct"] = (agg[col] / agg["Total"] * 100).round(1)

                    st.dataframe(agg[["aspect", "Positive_pct", "Neutral_pct", "Negative_pct"]])

                    # Plot stacked bar of percentages
                    fig = go.Figure()
                    for sentiment, color in [
                        ("Negative_pct", COLORS["negative"]),
                        ("Neutral_pct", COLORS["neutral"]),
                        ("Positive_pct", COLORS["positive"]),
                    ]:
                        fig.add_trace(go.Bar(
                            x=agg["aspect"],
                            y=agg[sentiment],
                            name=sentiment.replace("_pct", ""),
                            marker_color=color
                        ))

                    fig.update_layout(
                        barmode="stack",
                        xaxis_title="Aspect",
                        yaxis_title="Percentage of Clauses (%)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=450,
                    )

                    st.plotly_chart(fig, width="stretch")

                    # ---- Downloadable outputs ----
                    st.markdown("### ⬇️ Download Results")

                    csv_reviews = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Review-Level Sentiment CSV",
                        data=csv_reviews,
                        file_name="review_level_sentiment.csv",
                        mime="text/csv",
                    )

                    csv_aspects = aspects_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Aspect-Level (Clause) CSV",
                        data=csv_aspects,
                        file_name="aspect_level_sentiment.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Error during batch ABSA: {e}")

# --- Page: Manufacturer Report ---
def render_report():
    st.markdown("## 📋 Manufacturer Feedback Report")
    
    report_path = "outputs/manufacturer_recommendations.md"
    
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report_content = f.read()
        
        # Render the report using native markdown for best compatibility
        st.markdown("""
        <style>
        .report-text {
            color: #1D1D1F !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(report_content)
    else:
        st.warning("⚠️ Report not found.")
        st.info("Please run the `sentiment_pipeline.py` script to generate the analysis first.")

# ═══════════════════════════════════════════════════════════════════════════
# MANUFACTURER ANALYTICS HUB — helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _mfg_clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s,.]', '', text)
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([!?.:])\1+', r'\1', text)
    return text

def _mfg_is_meaningless(text: str) -> bool:
    if not text or not isinstance(text, str) or not text.strip():
        return True
    if set(text.lower().split()).issubset(MFG_FILLER):
        return True
    if sum(c.isalpha() for c in text) < 3:
        return True
    alnum = sum(c.isalnum() for c in text)
    if len(text) > 0 and alnum / len(text) < 0.3:
        return True
    return False

def _mfg_translate(text: str) -> str:
    if not MFG_TRANS_AVAILABLE or not text:
        return text
    try:
        lang = lang_detect(text)
        if lang not in ("en", "unknown"):
            return GoogleTranslator(source="auto", target="en").translate(text) or text
    except Exception:
        pass
    return text

def _mfg_detect_aspect(text: str) -> str:
    t = text.lower()
    best, best_score = "Other", 0
    for aspect, kws in MFG_ASPECT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best_score, best = score, aspect
    return best

def _mfg_map_label(raw: str) -> str:
    u = raw.upper()
    if any(x in u for x in ["5","4","POS","POSITIVE"]): return "Positive"
    if any(x in u for x in ["1","2","NEG","NEGATIVE"]): return "Negative"
    return "Neutral"

def _mfg_split_clauses(text: str):
    parts = re.split(r'(?i)\bbut\b|\bhowever\b|\bthough\b|[.!?]', text)
    return [p.strip() for p in parts if p.strip()]

@st.cache_resource(show_spinner=False)
def _mfg_load_classifier():
    if not TRANSFORMERS_AVAILABLE:
        return None
    return pipeline(
        "sentiment-analysis",
        model="unknownexplosion/SentimentAnalysisog",
        device=-1,
        model_kwargs={"low_cpu_mem_usage": False},
    )

def _mfg_run_sentiment(texts: list, clf) -> list:
    if clf is None:
        return [("Neutral", 0.5)] * len(texts)
    results = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        try:
            res = clf(batch, truncation=True, max_length=512)
            for r in res:
                results.append((_mfg_map_label(r["label"]), round(r["score"], 4)))
        except Exception:
            results.extend([("Neutral", 0.5)] * len(batch))
    return results

def _mfg_synthetic_weeks(n: int, rpw: int = 25) -> list:
    return [f"Week {(i // rpw) + 1:02d}" for i in range(n)]

def _mfg_run_pipeline(raw_df, model_col, review_col, date_col, progress_bar, status_text):
    df = raw_df[[model_col, review_col] + ([date_col] if date_col else [])].copy()
    df.columns = ["model", "original_review"] + (["date"] if date_col else [])
    df["model"] = df["model"].astype(str).str.strip()

    status_text.markdown("**Step 1/4** — Cleaning text…")
    progress_bar.progress(10)
    df["cleaned"] = df["original_review"].apply(_mfg_clean_text)
    df.loc[df["cleaned"].apply(_mfg_is_meaningless), "cleaned"] = np.nan

    status_text.markdown("**Step 2/4** — Translating non-English reviews…")
    progress_bar.progress(25)
    if MFG_TRANS_AVAILABLE:
        df["final"] = df["cleaned"].apply(lambda x: _mfg_translate(x) if pd.notna(x) else np.nan)
        df["final"] = df["final"].apply(lambda x: _mfg_clean_text(x) if pd.notna(x) else np.nan)
    else:
        df["final"] = df["cleaned"]

    df["_norm"] = df["final"].astype(str).str.lower().str.strip()
    df.loc[df.duplicated(subset=["model","_norm"], keep="first"), "final"] = np.nan
    df.drop(columns=["_norm"], inplace=True)

    status_text.markdown("**Step 3/4** — Extracting aspects (ABSA)…")
    progress_bar.progress(45)
    clf = _mfg_load_classifier()
    valid_mask = df["final"].notna() & (df["final"] != "")
    valid_texts = df.loc[valid_mask, "final"].tolist()
    sentiments = _mfg_run_sentiment(valid_texts, clf)
    df.loc[valid_mask, "sentiment_label"]      = [s[0] for s in sentiments]
    df.loc[valid_mask, "sentiment_confidence"] = [s[1] for s in sentiments]

    status_text.markdown("**Step 4/4** — Running sentiment model (aspect-conditioned)…")
    progress_bar.progress(75)
    absa_rows = []

    if SPACY_AVAILABLE:
        # Build a flat keyword → canonical aspect name lookup (from MFG_ASPECT_KEYWORDS)
        _kw_to_aspect = {}
        for _asp, _kws in MFG_ASPECT_KEYWORDS.items():
            for _kw in _kws:
                if _kw not in _kw_to_aspect:   # first definition wins
                    _kw_to_aspect[_kw] = _asp

        for _, row in df[valid_mask].iterrows():
            text  = row["final"]
            model = row["model"]
            doc   = _spacy_nlp(text)

            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                t_lower = sent_text.lower()

                # Find every aspect whose keywords appear in this sentence
                matched = {}
                for kw, asp in _kw_to_aspect.items():
                    if kw in t_lower and asp not in matched:
                        matched[asp] = True

                if not matched:
                    continue  # sentence mentions no tracked aspect → skip

                # Run the sentiment model ONCE per sentence (not per aspect)
                if clf:
                    try:
                        res     = clf([sent_text], truncation=True, max_length=512)[0]
                        s_label = _mfg_map_label(res["label"])
                        s_conf  = round(res["score"], 4)
                    except Exception:
                        s_label = row.get("sentiment_label", "Neutral") or "Neutral"
                        s_conf  = 0.5
                else:
                    s_label = row.get("sentiment_label", "Neutral") or "Neutral"
                    s_conf  = 0.5

                # Emit one row per (sentence, aspect) pair
                for asp in matched:
                    absa_rows.append({
                        "model":      model,
                        "text":       sent_text,
                        "aspect":     asp,
                        "label":      s_label,
                        "confidence": s_conf,
                    })
    else:
        # ── Fallback: original clause-split + keyword matching ──────────────
        for _, row in df[valid_mask].iterrows():
            text    = row["final"]
            model   = row["model"]
            clauses = _mfg_split_clauses(text) or [text]
            c_results = (
                _mfg_run_sentiment(clauses, clf)
                if clf
                else [(row.get("sentiment_label", "Neutral") or "Neutral", 0.5)] * len(clauses)
            )
            for clause, (s_label, s_conf) in zip(clauses, c_results):
                absa_rows.append({
                    "model":      model,
                    "text":       clause,
                    "aspect":     _mfg_detect_aspect(clause),
                    "label":      s_label,
                    "confidence": s_conf,
                })

    if date_col and "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["week"] = df["date"].dt.to_period("W").astype(str)
        except Exception:
            df["week"] = _mfg_synthetic_weeks(len(df))
    else:
        df["week"] = _mfg_synthetic_weeks(len(df))

    progress_bar.progress(100)
    status_text.markdown("✅ **Pipeline complete!**")
    return df, pd.DataFrame(absa_rows)


def _mfg_line_chart(df, model):
    df_f = df[(df["model"] == model) & df["sentiment_label"].notna()]
    rows = []
    for week, g in df_f.groupby("week"):
        total = len(g)
        rows.append({
            "Week":     week,
            "Positive": round((g["sentiment_label"]=="Positive").sum()/total*100,1),
            "Negative": round((g["sentiment_label"]=="Negative").sum()/total*100,1),
            "Neutral":  round((g["sentiment_label"]=="Neutral").sum()/total*100,1),
            "Count":    total,
        })
    wdf = pd.DataFrame(rows).sort_values("Week")
    fig = go.Figure()
    _FILL_ALPHA = {
        COLORS["positive"]: "rgba(52,199,89,0.09)",
        COLORS["negative"]: "rgba(255,59,48,0.09)",
        COLORS["neutral"]:  "rgba(142,142,147,0.09)",
    }
    for sentiment, color in [("Positive",COLORS["positive"]),("Neutral",COLORS["neutral"]),("Negative",COLORS["negative"])]:
        fig.add_trace(go.Scatter(
            x=wdf["Week"], y=wdf[sentiment], name=sentiment,
            mode="lines+markers",
            line=dict(color=color, width=2.5, shape="spline"),
            marker=dict(size=6, color=color),
            fill="tozeroy",
            fillcolor=_FILL_ALPHA.get(color, "rgba(142,142,147,0.09)"),
            hovertemplate=f"<b>%{{x}}</b><br>{sentiment}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=wdf["Week"], y=wdf["Count"], name="# Reviews",
        mode="lines", line=dict(color=COLORS["primary"], width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Reviews: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **MFG_PLOTLY_LAYOUT,
        height=370,
        xaxis=dict(title="", **MFG_GRID),
        yaxis=dict(title="Sentiment %", range=[0,105], **MFG_GRID),
        yaxis2=dict(title="# Reviews", overlaying="y", side="right", showgrid=False,
                    range=[0, wdf["Count"].max()*3] if len(wdf) else [0,10]),
        legend=MFG_LEGEND,
        margin=dict(t=20,b=40,l=50,r=50),
    )
    return fig


def _mfg_aspect_line_chart(absa_df, model):
    df_m = absa_df[absa_df["model"]==model].copy().reset_index(drop=True)
    if df_m.empty:
        return None
    df_m["week"] = [f"W{(i//20)+1:02d}" for i in range(len(df_m))]
    top_aspects = df_m["aspect"].value_counts().head(6).index.tolist()
    df_m = df_m[df_m["aspect"].isin(top_aspects)]
    palette = [COLORS["primary"],COLORS["positive"],"#FF9F0A","#5856D6","#FF6CAB","#5AC8FA"]
    fig = go.Figure()
    for i, asp in enumerate(top_aspects):
        wdf = (df_m[df_m["aspect"]==asp]
               .groupby("week")
               .apply(lambda g: round((g["label"]=="Positive").sum()/len(g)*100,1), include_groups=False)
               .reset_index(name="pos_pct"))
        fig.add_trace(go.Scatter(
            x=wdf["week"], y=wdf["pos_pct"], name=asp,
            mode="lines+markers",
            line=dict(color=palette[i%len(palette)], width=2, shape="spline"),
            marker=dict(size=5),
            hovertemplate=f"<b>%{{x}}</b><br>{asp} Positive: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        **MFG_PLOTLY_LAYOUT, height=330,
        xaxis=dict(**MFG_GRID),
        yaxis=dict(title="Positive %", range=[0,105], **MFG_GRID),
        legend=MFG_LEGEND,
        margin=dict(t=20,b=40,l=50,r=50),
    )
    return fig


def _mfg_radar_chart(absa_df, model):
    df_m = absa_df[absa_df["model"]==model]
    if df_m.empty:
        return None
    agg = (df_m.groupby("aspect")
           .apply(lambda g: round((g["label"]=="Positive").sum()/len(g)*100,1), include_groups=False)
           .reset_index(name="pos_pct")
           .sort_values("aspect"))
    cats  = agg["aspect"].tolist()
    vals  = agg["pos_pct"].tolist()
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor="rgba(0,122,255,0.15)",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(color=COLORS["primary"], size=6),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(255,255,255,0.04)",
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(color="#8E8E93",size=9),
                            gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(tickfont=dict(color="#C7C7CC",size=10),
                             gridcolor="rgba(255,255,255,0.1)"),
        ),
        height=310, margin=dict(t=20,b=20,l=40,r=40),
        font=dict(family="Inter",color="#C7C7CC"),
    )
    return fig


def _mfg_kpi_cards(df, absa_df, model):
    mdf   = df[df["model"]==model]
    valid = mdf[mdf["sentiment_label"].notna()]
    total = len(mdf)
    pos_p = round((valid["sentiment_label"]=="Positive").mean()*100,1) if len(valid) else 0
    neg_p = round((valid["sentiment_label"]=="Negative").mean()*100,1) if len(valid) else 0
    m_absa = absa_df[absa_df["model"]==model]
    top_pos_asp = top_neg_asp = "—"
    if not m_absa.empty:
        pos_asp = m_absa[m_absa["label"]=="Positive"]["aspect"].value_counts()
        neg_asp = m_absa[m_absa["label"]=="Negative"]["aspect"].value_counts()
        top_pos_asp = pos_asp.idxmax() if not pos_asp.empty else "—"
        top_neg_asp = neg_asp.idxmax() if not neg_asp.empty else "—"
    c1,c2,c3,c4,c5 = st.columns(5)
    cards = [
        (c1,"📦 Total Reviews",f"{total:,}","analysed"),
        (c2,"😊 Positive",f"{pos_p}%","of reviews"),
        (c3,"😤 Negative",f"{neg_p}%","of reviews"),
        (c4,"⭐ Top Strength",top_pos_asp,"most praised"),
        (c5,"⚠️ Top Issue",top_neg_asp,"most criticised"),
    ]
    for col, label, value, sub in cards:
        with col:
            st.markdown(f"""
            <div class="mfg-kpi">
                <div class="mfg-kpi-label">{label}</div>
                <div class="mfg-kpi-value">{value}</div>
                <div class="mfg-kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)


def _mfg_textual_feedback(df, absa_df, model):
    mdf = df[df["model"]==model]
    valid = mdf[mdf["sentiment_label"].notna()]
    if valid.empty:
        return {}
    pos_p = round((valid["sentiment_label"]=="Positive").mean()*100,1)
    neg_p = round((valid["sentiment_label"]=="Negative").mean()*100,1)
    neu_p = round((valid["sentiment_label"]=="Neutral").mean()*100,1)
    m_absa = absa_df[absa_df["model"]==model]
    strengths, issues = [], []
    if not m_absa.empty:
        for asp in m_absa["aspect"].unique():
            g = m_absa[m_absa["aspect"]==asp]
            pos = (g["label"]=="Positive").mean()
            neg = (g["label"]=="Negative").mean()
            if pos >= 0.6: strengths.append((asp, round(pos*100)))
            elif neg >= 0.4: issues.append((asp, round(neg*100)))
    tone = ("overwhelmingly positive" if pos_p>75 else
            "generally positive"      if pos_p>50 else
            "mixed"                   if pos_p>30 else "largely negative")
    return {"tone":tone,"pos_p":pos_p,"neg_p":neg_p,"neu_p":neu_p,
            "total":len(valid),
            "strengths":sorted(strengths,key=lambda x:-x[1])[:5],
            "issues":sorted(issues,key=lambda x:-x[1])[:5]}


def _mfg_render_textual_feedback(fb, model):
    if not fb:
        st.warning("Not enough data to generate feedback.")
        return
    tone_color = (COLORS["positive"] if "positive" in fb["tone"] else
                  COLORS["negative"] if "negative" in fb["tone"] else COLORS["neutral"])
    st.markdown(f"""
    <div class="mfg-glass">
        <div class="mfg-sh">📋 Automated Analysis</div>
        <p style="color:#C7C7CC;line-height:1.7;">
            Based on <b style="color:#F5F5F7">{fb['total']:,}</b> reviews,
            sentiment for <b style="color:#F5F5F7">{model}</b> is
            <b style="color:{tone_color}">{fb['tone']}</b>. &nbsp;
            <span class="mfg-pill-pos">Positive {fb['pos_p']}%</span>
            <span class="mfg-pill-neu">Neutral {fb['neu_p']}%</span>
            <span class="mfg-pill-neg">Negative {fb['neg_p']}%</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
        st.markdown("#### ✅ Top Strengths")
        if fb["strengths"]:
            for asp, pct in fb["strengths"]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:7px 0;
                            border-bottom:1px solid rgba(255,255,255,0.06);">
                    <span style="color:#C7C7CC">{asp}</span>
                    <span style="color:{COLORS['positive']};font-weight:700">{pct}% positive</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.caption("No dominant strengths detected.")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Areas to Improve")
        if fb["issues"]:
            for asp, pct in fb["issues"]:
                severity  = "HIGH" if pct>=60 else "MEDIUM" if pct>=40 else "LOW"
                sev_color = COLORS["negative"] if severity=="HIGH" else "#FF9F0A" if severity=="MEDIUM" else COLORS["neutral"]
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:7px 0;
                            border-bottom:1px solid rgba(255,255,255,0.06);">
                    <span style="color:#C7C7CC">{asp}</span>
                    <span style="color:{sev_color};font-weight:700">[{severity}] {pct}% negative</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.caption("No major issues detected.")
        st.markdown("</div>", unsafe_allow_html=True)


def _mfg_ai_feedback(model, absa_df):
    g_key = os.getenv("GOOGLE_API_KEY","")
    try:
        s = toml.load(".streamlit/secrets.toml")
        g_key = g_key or s.get("GOOGLE_API_KEY") or s.get("general",{}).get("GOOGLE_API_KEY","")
    except Exception:
        pass
    if not g_key:
        st.error("🔑 Google API key not found in `.streamlit/secrets.toml`.")
        return None
    try:
        from genai_bi import BISummarizer
    except ImportError as e:
        st.error(f"Could not import genai_bi.py: {e}")
        return None
    df_model = absa_df[absa_df["model"]==model]
    if df_model.empty:
        st.warning("No ABSA records for this model.")
        return None
    bot = BISummarizer()
    with st.status("🤖 Generating AI Report…", expanded=True) as status:
        st.write("Analysing sentiment patterns…")
        result = bot.generate_for_model(model, df_model)
        if result:
            status.update(label="✅ Report Generated!", state="complete")
            return result
        status.update(label="❌ Generation Failed", state="error")
        return None


def _mfg_render_ai(summary_json):
    bs = summary_json.get("business_summary",{})
    st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
    st.markdown("#### 📝 Executive Overview")
    st.info(bs.get("executive_overview","No overview."))
    st.markdown("</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
        st.markdown("#### ✅ Key Strengths")
        for item in bs.get("key_strengths",[]):
            with st.expander(f"**{item.get('aspect','Feature')}**", expanded=True):
                st.markdown(item.get("summary",""))
                s = item.get("supporting_sentiment",{})
                st.caption(f"Positive: {s.get('positive_share','N/A')} · Negative: {s.get('negative_share','N/A')}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Key Issues")
        for item in bs.get("key_issues",[]):
            priority = item.get("priority","MEDIUM")
            p_color  = "red" if priority=="HIGH" else "orange" if priority=="MEDIUM" else "blue"
            with st.expander(f"**{item.get('aspect','Feature')}** :{p_color}[{priority}]", expanded=True):
                st.markdown(item.get("summary",""))
                s = item.get("supporting_sentiment",{})
                st.caption(f"Negative: {s.get('negative_share','N/A')} · Positive: {s.get('positive_share','N/A')}")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("#### 🚀 Recommendations")
    for rec in bs.get("recommendations",[]):
        st.markdown(f"""
        <div style="background:rgba(0,122,255,0.08);padding:15px 18px;border-radius:13px;
                    border-left:4px solid {COLORS['primary']};margin-bottom:9px;">
            <h4 style="margin:0;color:#F5F5F7">{rec.get('title','Recommendation')}</h4>
            <p style="margin:5px 0 3px;color:#C7C7CC">{rec.get('description','')}</p>
            <p style="font-size:0.8rem;color:{COLORS['primary']};margin:0">
                <b>Expected Impact:</b> {rec.get('expected_impact','')}</p>
        </div>""", unsafe_allow_html=True)


def _mfg_model_section(model, review_df, absa_df):
    # KPI
    st.markdown(f"<div class='mfg-sh' style='margin-top:18px'><span class='mfg-step-badge'>2</span> Key Metrics — {model}</div>", unsafe_allow_html=True)
    _mfg_kpi_cards(review_df, absa_df, model)

    # Weekly line chart
    st.markdown("<div class='mfg-sh' style='margin-top:26px'><span class='mfg-step-badge'>3</span> Weekly Sentiment Trends</div>", unsafe_allow_html=True)
    st.markdown("<div class='mfg-sub'>Positive / Neutral / Negative % over time (synthetic weekly buckets if no date column).</div>", unsafe_allow_html=True)
    st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
    st.markdown("##### Overall Sentiment Over Time")
    st.plotly_chart(_mfg_line_chart(review_df, model), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
        st.markdown("##### Aspect Positive % Over Time")
        fig_asp = _mfg_aspect_line_chart(absa_df, model)
        if fig_asp: st.plotly_chart(fig_asp, width="stretch")
        else: st.caption("Not enough aspect data.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
        st.markdown("##### Aspect Satisfaction Radar")
        fig_r = _mfg_radar_chart(absa_df, model)
        if fig_r: st.plotly_chart(fig_r, width="stretch")
        else: st.caption("Not enough data.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Textual feedback
    st.markdown("<div class='mfg-sh' style='margin-top:26px'><span class='mfg-step-badge'>4</span> Automated Feedback</div>", unsafe_allow_html=True)
    _mfg_render_textual_feedback(_mfg_textual_feedback(review_df, absa_df, model), model)

    # AI feedback
    st.markdown("<div class='mfg-sh' style='margin-top:26px'><span class='mfg-step-badge'>5</span> AI Executive Report (Gemini)</div>", unsafe_allow_html=True)
    st.markdown("<div class='mfg-sub'>Click to generate an AI-powered executive summary for this model.</div>", unsafe_allow_html=True)
    cache_key = f"mfg_ai_{model}"
    if cache_key in st.session_state.get("mfg_ai_cache",{}):
        _mfg_render_ai(st.session_state["mfg_ai_cache"][cache_key])
    else:
        if st.button(f"✨ Generate AI Report for {model}", key=f"mfg_ai_{model}"):
            result = _mfg_ai_feedback(model, absa_df)
            if result:
                if "mfg_ai_cache" not in st.session_state:
                    st.session_state["mfg_ai_cache"] = {}
                st.session_state["mfg_ai_cache"][cache_key] = result
                st.rerun()

    # Downloads
    st.markdown("---")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(
            "📥 Download Sentiment CSV",
            data=review_df[review_df["model"]==model].to_csv(index=False).encode("utf-8"),
            file_name=f"{model.replace(' ','_')}_sentiment.csv",
            mime="text/csv", width="stretch",
        )
    with dc2:
        st.download_button(
            "📥 Download ABSA CSV",
            data=absa_df[absa_df["model"]==model].to_csv(index=False).encode("utf-8"),
            file_name=f"{model.replace(' ','_')}_absa.csv",
            mime="text/csv", width="stretch",
        )


# ── Main page renderer ─────────────────────────────────────────────────────
def render_manufacturer_hub():
    # Session state init
    for k in ("mfg_review_df","mfg_absa_df","mfg_done","mfg_ai_cache"):
        if k not in st.session_state:
            st.session_state[k] = False if k=="mfg_done" else ({} if k=="mfg_ai_cache" else None)

    # Hero
    st.markdown("""
    <div class="mfg-hero">
        <h1>📊 Manufacturer Analytics Hub</h1>
        <p>Upload any product review CSV → AI pipeline → weekly trends → insights</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: Upload ──────────────────────────────────────────────────────
    st.markdown("<div class='mfg-sh'><span class='mfg-step-badge'>1</span> Upload Review Data</div>", unsafe_allow_html=True)
    st.markdown("<div class='mfg-sub'>CSV must have a product/model column and a review/text column. Date column is optional.</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
        st.markdown(f"**Loaded `{uploaded.name}`** — {len(raw_df):,} rows · {len(raw_df.columns)} columns")
        st.dataframe(raw_df.head(5), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        cols = raw_df.columns.tolist()
        st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
        st.markdown("**Map your columns**")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            model_col  = st.selectbox("🏷️ Product / Model column", cols, key="mfg_model_col")
        with cc2:
            review_col = st.selectbox("💬 Review / Text column",   cols, key="mfg_review_col")
        with cc3:
            date_sel   = st.selectbox("📅 Date column (optional)", ["(none)"]+cols, key="mfg_date_col")
            date_col   = None if date_sel=="(none)" else date_sel
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Run Full Analysis", width="stretch", key="mfg_run"):
            st.session_state["mfg_done"]     = False
            st.session_state["mfg_ai_cache"] = {}
            progress_bar = st.progress(0)
            status_text  = st.empty()
            with st.spinner(""):
                r_df, a_df = _mfg_run_pipeline(raw_df, model_col, review_col, date_col, progress_bar, status_text)
            st.session_state["mfg_review_df"] = r_df
            st.session_state["mfg_absa_df"]   = a_df
            st.session_state["mfg_done"]       = True
            st.rerun()

    # ── Results ─────────────────────────────────────────────────────────────
    if not st.session_state["mfg_done"] or st.session_state["mfg_review_df"] is None:
        if not uploaded:
            st.markdown("""
            <div style="text-align:center;padding:55px 0;color:#8E8E93">
                <div style="font-size:2.8rem;margin-bottom:12px">📂</div>
                <div style="font-size:1.05rem">Upload a CSV above to get started</div>
                <div style="font-size:0.82rem;margin-top:6px">Supports any product reviews — not just Apple</div>
            </div>
            """, unsafe_allow_html=True)
        return

    review_df = st.session_state["mfg_review_df"]
    absa_df   = st.session_state["mfg_absa_df"]
    models    = sorted(review_df["model"].unique().tolist())

    st.success(f"✅ Pipeline complete — {len(review_df):,} reviews · {len(absa_df):,} aspect clauses · {len(models)} product(s)")
    st.markdown("---")

    if len(models) <= 6:
        tabs = st.tabs([f"**{m}**" for m in models])
        for model, tab in zip(models, tabs):
            with tab:
                _mfg_model_section(model, review_df, absa_df)
    else:
        sel_model = st.selectbox("Select Product Model", models, key="mfg_model_sel")
        _mfg_model_section(sel_model, review_df, absa_df)


# ═══════════════════════════════════════════════════════════════════════════
# REDDIT MODEL SCOUT
# ═══════════════════════════════════════════════════════════════════════════

def _scout_check_creds():
    """Return Reddit credentials dict or None if not configured."""
    creds = {
        "client_id":     os.getenv("REDDIT_CLIENT_ID", ""),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
        "user_agent":    os.getenv("REDDIT_USER_AGENT", "SentimentScoutBot/1.0"),
    }
    try:
        secrets = toml.load(".streamlit/secrets.toml")
        r = secrets.get("reddit", {})
        creds["client_id"]     = creds["client_id"]     or r.get("client_id", "")
        creds["client_secret"] = creds["client_secret"] or r.get("client_secret", "")
        creds["user_agent"]    = creds["user_agent"]    or r.get("user_agent", "SentimentScoutBot/1.0")
    except Exception:
        pass
    return creds if (creds["client_id"] and creds["client_secret"]) else None


def _scout_scrape_public(query: str, max_posts: int, time_filter: str, extra_subs: list) -> pd.DataFrame:
    """
    Credential-free scraper using Reddit's public JSON endpoint.
    No API keys required — works out of the box.
    """
    import urllib.request
    import urllib.parse
    import json as _json
    from datetime import datetime, timezone

    model_name = query.strip()
    rows       = []
    seen_ids   = set()
    HEADERS    = {"User-Agent": "SentimentScout/1.0 (public data reader)"}

    def _fetch_json(url: str) -> dict:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _json.loads(resp.read().decode())

    def _add_from_listing(children: list):
        for child in children:
            d = child.get("data", {})
            pid = d.get("id", "")
            if not pid or pid in seen_ids:
                continue
            title    = d.get("title", "")
            selftext = d.get("selftext", "") or ""
            text     = (title + " " + selftext).strip()
            if len(text) < 20:
                continue
            created = d.get("created_utc", 0)
            rows.append({
                "model_name": model_name,
                "review":     text,
                "date":       datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d"),
            })
            seen_ids.add(pid)

            # Fetch top comments for this post
            try:
                c_url = f"https://www.reddit.com/comments/{pid}.json?limit=5"
                c_data = _fetch_json(c_url)
                if isinstance(c_data, list) and len(c_data) > 1:
                    for cc in c_data[1].get("data", {}).get("children", []):
                        cd = cc.get("data", {})
                        body = (cd.get("body") or "").strip()
                        cid  = f"{pid}_c{cd.get('id','')}"
                        if len(body) >= 20 and body not in ("[deleted]", "[removed]") and cid not in seen_ids:
                            rows.append({
                                "model_name": model_name,
                                "review":     body,
                                "date":       datetime.fromtimestamp(
                                                  cd.get("created_utc", 0), tz=timezone.utc
                                              ).strftime("%Y-%m-%d"),
                            })
                            seen_ids.add(cid)
                time.sleep(0.4)  # polite rate limit between comment fetches
            except Exception:
                pass

    # --- Search r/all via public API ---
    queries_to_run = [
        f"{query} review",
        query,
    ]
    for q in queries_to_run:
        try:
            encoded = urllib.parse.quote_plus(q)
            url = (
                f"https://www.reddit.com/search.json"
                f"?q={encoded}&sort=relevance&t={time_filter}&limit={min(max_posts, 100)}"
            )
            data = _fetch_json(url)
            _add_from_listing(data.get("data", {}).get("children", []))
            time.sleep(1.0)  # respect rate limit
        except Exception as e:
            st.warning(f"Reddit public API error for query '{q}': {e}")

    # --- Extra subreddits (if provided) ---
    for sr in extra_subs:
        try:
            encoded = urllib.parse.quote_plus(query)
            url = (
                f"https://www.reddit.com/r/{sr}/search.json"
                f"?q={encoded}&restrict_sr=1&sort=relevance&t={time_filter}&limit={min(max_posts // 2, 50)}"
            )
            data = _fetch_json(url)
            _add_from_listing(data.get("data", {}).get("children", []))
            time.sleep(1.0)
        except Exception:
            pass

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["model_name", "review", "date"])


def _scout_scrape(query: str, max_posts: int, time_filter: str, extra_subs: list) -> pd.DataFrame:
    """
    Dual-mode scraper:
      1. Uses PRAW (Reddit OAuth) if credentials are configured in secrets.toml.
      2. Falls back to Reddit's public JSON endpoint — no credentials needed.
    """
    creds = _scout_check_creds()
    if creds:
        # ── PRAW (authenticated, higher rate limits) ──
        try:
            import praw
            from datetime import datetime, timezone

            reddit = praw.Reddit(
                client_id=creds["client_id"],
                client_secret=creds["client_secret"],
                user_agent=creds["user_agent"],
            )
            model_name = query.strip()
            rows, seen_ids = [], set()

            def _add_post(post):
                if post.id in seen_ids:
                    return
                text = (post.title + " " + post.selftext).strip()
                if len(text) < 20:
                    return
                rows.append({
                    "model_name": model_name,
                    "review":     text,
                    "date":       datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime("%Y-%m-%d"),
                })
                seen_ids.add(post.id)
                try:
                    post.comments.replace_more(limit=0)
                    for comment in list(post.comments)[:5]:
                        body = comment.body.strip()
                        if len(body) < 20 or body in ("[deleted]", "[removed]"):
                            continue
                        cid = f"{post.id}_c{comment.id}"
                        if cid not in seen_ids:
                            rows.append({
                                "model_name": model_name,
                                "review":     body,
                                "date":       datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).strftime("%Y-%m-%d"),
                            })
                            seen_ids.add(cid)
                except Exception:
                    pass

            for q in [f"{query} review", query]:
                try:
                    for post in reddit.subreddit("all").search(q, sort="relevance", time_filter=time_filter, limit=max_posts):
                        _add_post(post)
                    time.sleep(0.5)
                except Exception as e:
                    st.warning(f"PRAW search error: {e}")

            for sr_name in extra_subs:
                try:
                    for post in reddit.subreddit(sr_name).search(query, sort="relevance", time_filter=time_filter, limit=max_posts // 2):
                        _add_post(post)
                    time.sleep(0.3)
                except Exception:
                    pass

            return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["model_name", "review", "date"])

        except ImportError:
            pass  # PRAW not installed — fall through to public API
        except Exception as e:
            st.warning(f"PRAW failed ({e}), switching to public Reddit API…")

    # ── Public JSON fallback (no credentials needed) ──
    return _scout_scrape_public(query, max_posts, time_filter, extra_subs)


def render_reddit_scout():
    """Page: search any model → scrape Reddit → full sentiment pipeline → dashboard."""

    # ── Session state init ────────────────────────────────────────────────
    for k in ("scout_review_df", "scout_absa_df", "scout_done",
              "scout_ai_cache", "scout_query", "scout_csv"):
        if k not in st.session_state:
            st.session_state[k] = (
                False if k == "scout_done" else
                {}    if k == "scout_ai_cache" else
                None
            )

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0d1117 100%);
        border: 1px solid rgba(0,122,255,0.3);
        border-radius: 24px;
        padding: 38px 48px;
        margin-bottom: 28px;
        text-align: center;
    ">
        <div style="font-size:2.8rem;margin-bottom:8px">🔍</div>
        <h1 style="color:#FFFFFF;font-size:2.2rem;font-weight:800;margin:0;font-family:'Inter',sans-serif">
            Reddit Model Scout
        </h1>
        <p style="color:#8E8E93;font-size:1rem;margin:10px 0 0">
            Search any product — scrape Reddit reviews — run the full AI pipeline — get insights & line graphs
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Credential check (informational only — public API works without creds) ──
    creds_ok = _scout_check_creds() is not None
    if not creds_ok:
        st.markdown("""
        <div style="background:rgba(0,122,255,0.08);border:1px solid rgba(0,122,255,0.25);
                    border-radius:16px;padding:18px 24px;margin-bottom:20px">
            <span style="color:#007AFF;font-weight:700">ℹ️ Using Reddit Public API (no credentials needed)</span>
            <p style="color:#8E8E93;margin:8px 0 0;font-size:0.88rem">
                Scraping works immediately without any setup. Optionally, you can add Reddit API
                credentials to <code>.streamlit/secrets.toml</code> under <code>[reddit]</code>
                for higher rate limits and more results.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Search UI ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="color:#F5F5F7;font-size:1.2rem;font-weight:700;margin-bottom:6px;display:flex;align-items:center">
        <span style="background:linear-gradient(135deg,#007AFF,#5856D6);color:white;
                     border-radius:50%;width:26px;height:26px;display:inline-flex;
                     align-items:center;justify-content:center;font-size:0.75rem;
                     font-weight:700;margin-right:10px">1</span>
        Search Configuration
    </div>
    <div style="color:#8E8E93;font-size:0.88rem;margin-bottom:18px">
        Enter any product or model name to scrape & analyse Reddit opinions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)

    sc1, sc2 = st.columns([2, 1])
    with sc1:
        query = st.text_input(
            "🔍 Product / Model name",
            placeholder="e.g. Samsung Galaxy S24, OnePlus 12, Sony WH-1000XM5 …",
            key="scout_query_input",
        )
    with sc2:
        time_filter = st.selectbox(
            "📅 Time period",
            ["week", "month", "year", "all"],
            index=1,
            key="scout_time_filter",
        )

    ac1, ac2 = st.columns([1, 1])
    with ac1:
        max_posts = st.slider(
            "Max posts per search query",
            min_value=10, max_value=150, value=50, step=10,
            key="scout_max_posts",
        )
    with ac2:
        extra_subs_input = st.text_input(
            "Extra subreddits to search (comma-separated, optional)",
            placeholder="e.g. gadgets, Android, smartphones",
            key="scout_extra_subs",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Scrape button ─────────────────────────────────────────────────────
    if not query.strip():
        st.caption("Enter a product name above to get started.")

    run_clicked = st.button(
        "🚀 Scrape Reddit & Analyse",
        width="stretch",
        disabled=not query.strip(),
        key="scout_run_btn",
    )

    if run_clicked and query.strip():
        st.session_state["scout_done"]     = False
        st.session_state["scout_ai_cache"] = {}
        st.session_state["scout_query"]    = query.strip()

        extra_subs = [
            s.strip() for s in extra_subs_input.split(",")
            if s.strip()
        ]

        progress_bar = st.progress(0)
        status_text  = st.empty()

        # ── STEP 1: Scrape ──
        status_text.markdown("""
        <div style="color:#C7C7CC;margin:8px 0">
            <b style="color:#007AFF">Step 1/5</b> — Scraping Reddit for
            <b style="color:#F5F5F7">""" + query.strip() + """</b> reviews…
        </div>""", unsafe_allow_html=True)
        progress_bar.progress(10)

        with st.spinner("Connecting to Reddit API…"):
            scraped_df = _scout_scrape(
                query.strip(), max_posts, time_filter, extra_subs
            )

        if scraped_df.empty:
            progress_bar.progress(0)
            status_text.empty()
            st.error(
                f'No Reddit posts found for **"{query.strip()}"**. '
                "Try a different search term, wider time period, or more subreddits."
            )
        else:
            progress_bar.progress(20)
            st.success(f"✅ Scraped **{len(scraped_df):,}** reviews/comments from Reddit.")

            # Store CSV in session state for download
            st.session_state["scout_csv"] = scraped_df.to_csv(index=False).encode("utf-8")

            # ── STEPS 2-5: Pipeline ──
            # Re-use the existing _mfg_run_pipeline helper
            # but we offset progress to 20-100 range
            class _OffsetProgress:
                """Wrapper that maps [0-100] → [20-100] on the shared bar."""
                def __init__(self, bar): self._bar = bar
                def progress(self, v): self._bar.progress(20 + int(v * 0.80))

            offset_bar      = _OffsetProgress(progress_bar)
            status_text_mfg = st.empty()

            review_df, absa_df = _mfg_run_pipeline(
                scraped_df,
                model_col  = "model_name",
                review_col = "review",
                date_col   = "date",
                progress_bar = offset_bar,
                status_text  = status_text_mfg,
            )

            st.session_state["scout_review_df"] = review_df
            st.session_state["scout_absa_df"]   = absa_df
            st.session_state["scout_done"]       = True
            st.rerun()

    # ── Results ───────────────────────────────────────────────────────────
    if not st.session_state["scout_done"] or st.session_state["scout_review_df"] is None:
        if not run_clicked:
            st.markdown("""
            <div style="text-align:center;padding:60px 0;color:#8E8E93">
                <div style="font-size:3rem;margin-bottom:14px">🔍</div>
                <div style="font-size:1.1rem">Enter a product name and click <b>Scrape Reddit &amp; Analyse</b></div>
                <div style="font-size:0.85rem;margin-top:8px">
                    Works with any brand — Samsung, Sony, OnePlus, Dyson, Nike…
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    review_df  = st.session_state["scout_review_df"]
    absa_df    = st.session_state["scout_absa_df"]
    saved_q    = st.session_state.get("scout_query", "Unknown")
    models     = sorted(review_df["model"].unique().tolist())

    # ── Pipeline summary banner ───────────────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(52,199,89,0.1);border:1px solid rgba(52,199,89,0.3);
                border-radius:14px;padding:16px 22px;margin:18px 0">
        <span style="color:#34C759;font-weight:700">✅ Pipeline complete</span>
        &nbsp;·&nbsp;
        <span style="color:#C7C7CC">{len(review_df):,} reviews</span>
        &nbsp;·&nbsp;
        <span style="color:#C7C7CC">{len(absa_df):,} aspect clauses</span>
        &nbsp;·&nbsp;
        <span style="color:#C7C7CC">Query: <b style="color:#F5F5F7">{saved_q}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # ── CSV Download (Step 2) ─────────────────────────────────────────────
    st.markdown("""
    <div style="color:#F5F5F7;font-size:1.2rem;font-weight:700;margin:24px 0 8px;display:flex;align-items:center">
        <span style="background:linear-gradient(135deg,#007AFF,#5856D6);color:white;
                     border-radius:50%;width:26px;height:26px;display:inline-flex;
                     align-items:center;justify-content:center;font-size:0.75rem;
                     font-weight:700;margin-right:10px">2</span>
        Download Scraped CSV
    </div>
    <div style="color:#8E8E93;font-size:0.88rem;margin-bottom:14px">
        Three-column format: <code>model_name</code> | <code>review</code> | <code>date</code>
    </div>
    """, unsafe_allow_html=True)

    dl1, dl2, dl3 = st.columns([1, 1, 2])
    with dl1:
        if st.session_state.get("scout_csv") is not None:
            st.download_button(
                "📥 Download Raw CSV",
                data=st.session_state["scout_csv"],
                file_name=f"{saved_q.replace(' ', '_')}_reddit_reviews.csv",
                mime="text/csv",
                width="stretch",
                key="scout_dl_raw",
            )
    with dl2:
        st.download_button(
            "📥 Download Sentiment CSV",
            data=review_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{saved_q.replace(' ', '_')}_sentiment.csv",
            mime="text/csv",
            width="stretch",
            key="scout_dl_sent",
        )

    st.markdown("---")

    # ── Per-model results (usually only one model = the search query) ─────
    # Use tabs if multiple models somehow detected, else single container
    if len(models) <= 6:
        tabs = st.tabs([f"**{m}**" for m in models])
        for model, tab in zip(models, tabs):
            with tab:
                _scout_model_section(model, review_df, absa_df, saved_q)
    else:
        sel = st.selectbox("Select Model", models, key="scout_model_sel")
        _scout_model_section(sel, review_df, absa_df, saved_q)


def _scout_model_section(model, review_df, absa_df, query):
    """Renders the full analysis for one model inside the Scout page."""

    # ── KPI Cards (Step 3) ────────────────────────────────────────────────
    st.markdown(f"""
    <div style="color:#F5F5F7;font-size:1.2rem;font-weight:700;margin:24px 0 10px;display:flex;align-items:center">
        <span style="background:linear-gradient(135deg,#007AFF,#5856D6);color:white;
                     border-radius:50%;width:26px;height:26px;display:inline-flex;
                     align-items:center;justify-content:center;font-size:0.75rem;
                     font-weight:700;margin-right:10px">3</span>
        Key Performance Metrics — {model}
    </div>
    """, unsafe_allow_html=True)
    _mfg_kpi_cards(review_df, absa_df, model)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sentiment Snapshot (donut) ────────────────────────────────────────
    mdf   = review_df[(review_df["model"] == model) & review_df["sentiment_label"].notna()]
    if not mdf.empty:
        snap1, snap2 = st.columns([1, 2])
        with snap1:
            st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
            st.markdown("##### Sentiment Distribution")
            counts = mdf["sentiment_label"].value_counts().reset_index()
            counts.columns = ["Label", "Count"]
            fig_donut = go.Figure(go.Pie(
                labels=counts["Label"],
                values=counts["Count"],
                hole=0.62,
                marker=dict(colors=[
                    COLORS.get(lbl.lower(), "#8E8E93")
                    for lbl in counts["Label"]
                ]),
                textfont=dict(color="#C7C7CC"),
            ))
            fig_donut.update_layout(
                **MFG_PLOTLY_LAYOUT,
                height=260,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#C7C7CC"),
                ),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_donut, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with snap2:
            st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
            st.markdown("##### Aspect Coverage")
            m_absa = absa_df[absa_df["model"] == model]
            if not m_absa.empty:
                asp_counts = m_absa["aspect"].value_counts().reset_index()
                asp_counts.columns = ["Aspect", "Count"]
                palette = [
                    COLORS["primary"], COLORS["positive"], "#FF9F0A",
                    "#5856D6", "#FF6CAB", "#5AC8FA", "#FF3B30", "#34C759",
                    "#8E8E93", "#FFCC02", "#AF52DE",
                ]
                fig_bar = go.Figure(go.Bar(
                    x=asp_counts["Aspect"],
                    y=asp_counts["Count"],
                    marker_color=[palette[i % len(palette)] for i in range(len(asp_counts))],
                    text=asp_counts["Count"],
                    textposition="outside",
                    textfont=dict(color="#C7C7CC", size=10),
                ))
                fig_bar.update_layout(
                    **MFG_PLOTLY_LAYOUT,
                    height=260,
                    xaxis=dict(**MFG_GRID, tickfont=dict(size=10)),
                    yaxis=dict(**MFG_GRID),
                    margin=dict(t=10, b=40, l=40, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Weekly Sentiment Line Charts (Step 4) ─────────────────────────────
    st.markdown("""
    <div style="color:#F5F5F7;font-size:1.2rem;font-weight:700;margin:28px 0 6px;display:flex;align-items:center">
        <span style="background:linear-gradient(135deg,#007AFF,#5856D6);color:white;
                     border-radius:50%;width:26px;height:26px;display:inline-flex;
                     align-items:center;justify-content:center;font-size:0.75rem;
                     font-weight:700;margin-right:10px">4</span>
        Weekly Sentiment Trends
    </div>
    <div style="color:#8E8E93;font-size:0.88rem;margin-bottom:16px">
        Positive / Neutral / Negative % over time (real dates from Reddit post timestamps).
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
    st.markdown("##### Overall Sentiment Over Time")
    st.plotly_chart(_mfg_line_chart(review_df, model), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    lc1, lc2 = st.columns([3, 2])
    with lc1:
        st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
        st.markdown("##### Aspect Positive % Over Time")
        fig_asp = _mfg_aspect_line_chart(absa_df, model)
        if fig_asp:
            st.plotly_chart(fig_asp, width="stretch")
        else:
            st.caption("Not enough aspect data.")
        st.markdown("</div>", unsafe_allow_html=True)
    with lc2:
        st.markdown('<div class="mfg-glass">', unsafe_allow_html=True)
        st.markdown("##### Aspect Satisfaction Radar")
        fig_r = _mfg_radar_chart(absa_df, model)
        if fig_r:
            st.plotly_chart(fig_r, width="stretch")
        else:
            st.caption("Not enough data.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Automated Textual Feedback (Step 5) ───────────────────────────────
    st.markdown("""
    <div style="color:#F5F5F7;font-size:1.2rem;font-weight:700;margin:28px 0 6px;display:flex;align-items:center">
        <span style="background:linear-gradient(135deg,#007AFF,#5856D6);color:white;
                     border-radius:50%;width:26px;height:26px;display:inline-flex;
                     align-items:center;justify-content:center;font-size:0.75rem;
                     font-weight:700;margin-right:10px">5</span>
        Automated Business Feedback
    </div>
    """, unsafe_allow_html=True)
    _mfg_render_textual_feedback(_mfg_textual_feedback(review_df, absa_df, model), model)

    # ── AI Executive Report (Step 6) ─────────────────────────────────────
    st.markdown("""
    <div style="color:#F5F5F7;font-size:1.2rem;font-weight:700;margin:28px 0 6px;display:flex;align-items:center">
        <span style="background:linear-gradient(135deg,#007AFF,#5856D6);color:white;
                     border-radius:50%;width:26px;height:26px;display:inline-flex;
                     align-items:center;justify-content:center;font-size:0.75rem;
                     font-weight:700;margin-right:10px">6</span>
        AI-Generated Executive Report (Gemini)
    </div>
    <div style="color:#8E8E93;font-size:0.88rem;margin-bottom:16px">
        Executive overview · Key strengths · Issues · Actionable recommendations.
    </div>
    """, unsafe_allow_html=True)

    cache_key = f"scout_ai_{model}"
    ai_cache  = st.session_state.get("scout_ai_cache", {})
    if cache_key in ai_cache:
        _mfg_render_ai(ai_cache[cache_key])
    else:
        if st.button(f"✨ Generate AI Report for {model}", key=f"scout_ai_btn_{model}"):
            result = _mfg_ai_feedback(model, absa_df)
            if result:
                if "scout_ai_cache" not in st.session_state:
                    st.session_state["scout_ai_cache"] = {}
                st.session_state["scout_ai_cache"][cache_key] = result
                st.rerun()

    # ── ABSA detail table ─────────────────────────────────────────────────
    with st.expander("🔬 View ABSA clause-level detail"):
        m_absa = absa_df[absa_df["model"] == model][["text", "aspect", "label", "confidence"]]
        st.dataframe(m_absa.head(200), width="stretch")

    # ── Downloads ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⬇️ Download Full Results")
    bdl1, bdl2 = st.columns(2)
    with bdl1:
        st.download_button(
            "📥 Sentiment Results CSV",
            data=review_df[review_df["model"] == model].to_csv(index=False).encode("utf-8"),
            file_name=f"{model.replace(' ', '_')}_sentiment.csv",
            mime="text/csv",
            width="stretch",
            key=f"scout_dl_sentiment_{model}",
        )
    with bdl2:
        st.download_button(
            "📥 ABSA Aspect CSV",
            data=absa_df[absa_df["model"] == model].to_csv(index=False).encode("utf-8"),
            file_name=f"{model.replace(' ', '_')}_absa.csv",
            mime="text/csv",
            width="stretch",
            key=f"scout_dl_absa_{model}",
        )


def render_reddit_data():
    st.markdown("## 🗄️ Reddit Data Warehouse")
    st.markdown("View all the data scraped continuously by the Reddit Auto-Scraper.")
    
    csv_path = "outputs/scraped/reddit_reviews_all.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            st.metric("Total Extracted Reviews", f"{len(df):,}")
            st.dataframe(df, use_container_width=True)
            
            with open(csv_path, "rb") as file:
                st.download_button(
                    label="Download Full Dataset",
                    data=file,
                    file_name="reddit_reviews_all.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.warning(f"File not found: {csv_path}. The scraper might not have run yet.")


if __name__ == "__main__":
    main()
