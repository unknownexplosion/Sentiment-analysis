import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from fpdf import FPDF

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Transformers import failed: {e}")

import re

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

    # Sidebar Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Project Overview", "Live Dashboard", "Model Playground", "Manufacturer Report", "Strategy Hub"], label_visibility="collapsed")
        
        st.markdown("---")
        st.info("💡 **Tip:** Use the 'Playground' to test your own text.")
        st.markdown("---")
        st.caption("v2.1 • GenAI & MongoDB")

    if page == "Project Overview":
        render_overview()
    elif page == "Live Dashboard":
        render_dashboard(df, absa_df)
    elif page == "Model Playground":
        render_playground()
    elif page == "Manufacturer Report":
        render_report()
    elif page == "Strategy Hub":
        render_strategy_hub(df)


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
        HF_MODEL_NAME = "unknownexplosion/Anubhav"  # your HF model
        base_dir = os.getcwd()
        local_model_path = os.path.join(base_dir, "outputs", "fine_tuned_absa_model")

        if os.path.exists(local_model_path):
            final_model_name = local_model_path
            st.sidebar.success("Model: Local Fine-Tuned DeBERTa")
        else:
            final_model_name = HF_MODEL_NAME
            if "bert-base" in HF_MODEL_NAME:
                st.sidebar.warning("Model: Base (Not Fine-Tuned)")
            else:
                st.sidebar.info(f"Model: Cloud ({HF_MODEL_NAME})")

        with st.spinner("Loading sentiment model..."):
            classifier = pipeline("sentiment-analysis", model=final_model_name)
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


# --- Helper: PDF Generator ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'GenAI Strategy Report', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def create_pdf(model_name, report_text):
    pdf = PDFReport()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Manufacturer Report: {model_name}", new_x="LMARGIN", new_y="NEXT", align='L')
    pdf.ln(5)
    
    # Body
    pdf.set_font("Helvetica", size=11)
    
    # Handle basic markdown-like formatting for clearer PDF
    # Simple replacement for bolding since FPDF doesn't support markdown natively without plugins
    clean_text = report_text.replace("**", "").replace("###", "").replace("##", "")
    
    lines = clean_text.split('\n')
    for line in lines:
        try:
           encoded_line = line.strip()
           if encoded_line:
               pdf.multi_cell(0, 6, encoded_line)
           pdf.ln(1)
        except:
           pass
        
    return pdf.output(dest='S') # Return as byte string

# --- Page: GenAI Strategy Hub ---
def render_strategy_hub(df):
    st.markdown("## 🤖 GenAI Strategy & Data Hub")
    st.markdown("Leverage **Google Gemini** to generate strategic insights and store structured data in **MongoDB**.")
    
    # 1. Credentials Setup
    with st.expander("🔑 Configuration (API Keys)", expanded=True):
        col1, col2 = st.columns(2)
        
        # Load from secrets
        default_google = st.secrets.get("general", {}).get("GOOGLE_API_KEY", "")
        default_mongo = st.secrets.get("general", {}).get("MONGO_URI", "")

        # Logic: If secrets exist, don't show them in plain text boxes.
        # Just show a status indicator.
        
        with col1:
            if default_google:
                st.success("✅ Google API Key Loaded")
                google_key = default_google
            else:
                google_key = st.text_input("Google Gemini API Key", type="password", help="Get it from Google AI Studio")

        with col2:
            if default_mongo:
                st.success("✅ MongoDB URI Loaded")
                mongo_uri = default_mongo
            else:
                mongo_uri = st.text_input("MongoDB Connection URI", type="password", help="e.g., mongodb://localhost:27017/")

    if not df.empty:
        # 2. Select Data
        st.divider()
        st.subheader("1. Select Data used for Analysis")
        model_list = sorted(df['model'].dropna().unique().tolist())
        selected_model_analyze = st.selectbox("Choose a Model to Analyze", model_list)
        
        # Filter data
        model_df = df[df['model'] == selected_model_analyze]
        st.write(f"Found **{len(model_df)}** reviews for {selected_model_analyze}")
        
        if st.button("🚀 Run GenAI Analysis"):
            if not google_key:
                st.error("Please enter your Google API Key.")
                return
            
            # Initialize Module
            try:
                from genai_analysis import GenAIAnalyzer
                analyzer = GenAIAnalyzer(google_api_key=google_key, mongo_uri=mongo_uri)
                
                with st.status("🤖 AI Agent Working...", expanded=True) as status:
                    st.write("Constructing prompt context...")
                    # Limit sample size to avoid token limits for demo
                    sample_df = model_df.head(50) 
                    
                    st.write("Calling Gemini 2.0 Flash...")
                    result = analyzer.generate_report(selected_model_analyze, sample_df)
                    
                    if "error" in result:
                        status.update(label="❌ Analysis Failed", state="error")
                        st.error(result["error"])
                    else:
                        st.write("Parsing strategic insights...")
                        report = result.get("manufacturer_report", {}).get("report", "No report generated.")
                        
                        status.update(label="✅ Analysis Complete!", state="complete")
                        
                        if "manufacturer_report" in result:
                            report_content = result["manufacturer_report"]["report"]
                            st.subheader("📋 GenAI Manufacturer Report") # Added this back for consistency
                            st.markdown(report_content)
                            
                            # PDF Download Button
                            st.divider()
                            pdf_bytes = create_pdf(selected_model_analyze, report_content)
                            st.download_button(
                                label="📄 Download Report as PDF",
                                data=bytes(pdf_bytes),
                                file_name=f"{selected_model_analyze}_Strategy_Report.pdf",
                                mime="application/pdf"
                            )
                        
                        # Display Structured Data (moved outside the if/else for report display, as it's always shown)
                        st.subheader("💾 Structured Aspect Data")
                        records = result.get("review_aspect_records", [])
                        st.dataframe(pd.DataFrame(records))
                        
                        # 3. Store to DB
                        if mongo_uri:
                            with st.spinner("Saving to MongoDB..."):
                                success, msg = analyzer.save_to_db(result)
                                if success:
                                    st.success(f"✅ {msg}")
                                else:
                                    st.error(f"❌ {msg}")
                        else:
                            st.warning("⚠️ MongoDB URI not provided. Skipping database storage.")
                            
            except ImportError:
                st.error("Module 'genai_analysis' not found. Please check setup.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("No data found to analyze.")

if __name__ == "__main__":
    main()
