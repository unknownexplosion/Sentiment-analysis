import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
# from fpdf import FPDF # GenAI removed

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Transformers import failed: {e}")

import re
import time
import pymongo
import certifi
import toml

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

# Custom CSS for Minimalist Apple-like Design
st.markdown("""
    <style>
    .main {
        background-color: #FBFBFD; /* Apple Light Grey Background */
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
    .stMetricLabel {
        color: #86868B;
    }
    .stMetricValue {
        color: #1D1D1F;
    }
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

# --- Page logic ---
def main():
    if df is None:
        st.error("🚨 Data not found. Please run the pipeline first.")
        return

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Project Overview", "Live Dashboard", "Model Playground", "Manufacturer Report", "Business Intelligence"], label_visibility="collapsed")
        
        st.markdown("---")
        st.info("💡 **Tip:** Use the 'Playground' to test your own text.")
        st.markdown("---")
        st.caption("v2.2 • GenAI Summaries")

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
        client = pymongo.MongoClient(mongo_uri, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
        db = client.get_database("sentiment_analysis_db")
        col = db["manufacturer_bi_summaries"]
        
        # 2. Fetch Models
        models = col.distinct("model")
        
        if not models:
            st.warning("No BI reports found in database.")
            # Fallback to allow generation if absa_df exists
            if not absa_df.empty and 'model_name' in absa_df.columns:
                 models = sorted(absa_df['model_name'].dropna().unique().tolist())
            elif not absa_df.empty and 'model' in absa_df.columns: # Handle flexible column naming
                 models = sorted(absa_df['model'].dropna().unique().tolist())
            else:
                 st.error("No ABSA data found to generate reports.")
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
                    model_data = absa_df[absa_df[model_col] == selected_model].to_dict('records')
                    
                    if not model_data:
                        st.error("No ABSA data available for this model.")
                    else:
                        with st.status("🤖 AI Agent Generating Report...", expanded=True) as status:
                            bi_bot = BISummarizer()
                            status.write("Analyzing sentiment patterns...")
                            # Generate
                            summary_json = bi_bot.generate_for_model(selected_model, model_data[:100]) # Limit context
                            
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
        
        st.markdown("### 🏆 Model Performance")
        st.markdown(f"""
        Our fine-tuned **DeBERTa v3** model achieves industry-leading metrics:
        
        | Metric | Score |
        | :--- | :--- |
        | **Accuracy** | <span style='color:{COLORS['positive']}'>**91.5%**</span> |
        | **F1-Score** | 0.915 |
        | **Precision** | 0.916 |
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
        st.plotly_chart(fig_donut, use_container_width=True)

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
        st.plotly_chart(fig_bar, use_container_width=True)

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
        st.plotly_chart(fig_absa, use_container_width=True)

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
        HF_MODEL_NAME = "unknownexplosion/Anubhav"
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

    if st.button("Analyze Sentiment", use_container_width=True):
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
                    st.dataframe(clause_df, use_container_width=True)

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
                    st.dataframe(df, use_container_width=True)

                    # Create aspect-level DataFrame
                    aspects_df = pd.DataFrame(aspect_rows)
                    if aspects_df.empty:
                        st.info("No aspects detected in the uploaded text.")
                        return

                    st.markdown("### 🔍 Clause & Aspect-Level Details")
                    st.dataframe(aspects_df.head(100), use_container_width=True)

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

                    st.plotly_chart(fig, use_container_width=True)

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

if __name__ == "__main__":
    main()
