import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Transformers import failed: {e}")


# --- Configuration & Styling ---
st.set_page_config(
    page_title="Apple Sentiment Analysis",
    page_icon="assets/apple_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
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
        page = st.radio("Go to", ["Project Overview", "Live Dashboard", "Model Playground", "Manufacturer Report"], label_visibility="collapsed")
        
        st.markdown("---")
        st.info("💡 **Tip:** Use the 'Playground' to test your own text.")
        st.markdown("---")
        st.caption("v2.0 • DeBERTa Powered")

    if page == "Project Overview":
        render_overview()
    elif page == "Live Dashboard":
        render_dashboard(df, absa_df)
    elif page == "Model Playground":
        render_playground()
    elif page == "Manufacturer Report":
        render_report()

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
    st.markdown("<div style='text-align: center;'><h2>🧠 Model Playground</h2><p>Test the fine-tuned DeBERTa model with your own text.</p></div>", unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns([1, 2, 1])
    
    with col_b:
        user_input = st.text_area("Enter a review:", height=150, placeholder="e.g., The screen is beautiful but the battery life is disappointing.")
        
        if st.button("Analyze Sentiment", use_container_width=True):
            if not user_input.strip():
                st.warning("Please enter some text.")
            elif not TRANSFORMERS_AVAILABLE:
                st.error("⚠️ Transformers library could not be loaded. Please check your installation.")
            else:
                with st.spinner("Processing with DeBERTa..."):
                    try:
                        # Load Model Logic
                        # --------------------------------------------------------
                        # 1. Configuration: Replace with YOUR Hugging Face Model ID after running upload_to_hub.py
                        #    Example: HF_MODEL_NAME = "anubhavmukherjee/apple-absa-v1"
                        HF_MODEL_NAME = "unknownexplosion/Anubhav" # Cloud Model ID
                        
                        # 2. Check for Local Model (Dev Environment)
                        base_dir = os.getcwd()
                        local_model_path = os.path.join(base_dir, "outputs", "fine_tuned_absa_model")
                        
                        if os.path.exists(local_model_path):
                             final_model_name = local_model_path
                             st.sidebar.success(f"Model: Local Fine-Tuned")
                        else:
                             # 3. Check for Cloud Model (Deployment)
                             # If you uploaded your model, change HF_MODEL_NAME above!
                             final_model_name = HF_MODEL_NAME
                             if "bert-base" in HF_MODEL_NAME:
                                 st.sidebar.warning("Note: Using Base Model (Not Fine-Tuned)")
                             else:
                                 st.sidebar.info(f"Model: Cloud ({HF_MODEL_NAME})")

                        classifier = pipeline("sentiment-analysis", model=final_model_name)
                        result = classifier(user_input)[0]
                        
                        label = result['label']
                        score = result['score']
                        
                        # Map Star labels (from BERT base) to Words if using base model, 
                        # DeBERTa fine-tunes usually outputs Neutral/Positive/Negative directly if mapped that way.
                        # Assuming our pipeline outputs standard labels.
                        
                        # Normalize label color
                        color = COLORS['neutral']
                        if '5' in label or '4' in label or 'POS' in label.upper() or 'Positive' in label:
                            display_label = "Positive"
                            color = COLORS['positive']
                        elif '1' in label or '2' in label or 'NEG' in label.upper() or 'Negative' in label:
                            display_label = "Negative"
                            color = COLORS['negative']
                        else:
                            display_label = "Neutral"

                        st.markdown(f"""
                        <div style="background-color: {color}20; padding: 20px; border-radius: 12px; border: 2px solid {color}; text-align: center; margin-top: 20px;">
                            <h3 style="color: {color}; margin:0;">{display_label}</h3>
                            <p style="margin:0; font-weight:bold;">Confidence: {score:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if os.path.exists(local_model_path):
                            st.caption("✅ Processed locally using **Fine-Tuned DeBERTa**")
                        else:
                            st.caption("⚠️ Processed using **base model** (or Cloud Model)")

                    except Exception as e:
                        st.error(f"Error: {e}")

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
