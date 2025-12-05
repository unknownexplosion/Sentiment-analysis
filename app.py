import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Manufacturer Feedback & Sentiment Analysis System")

# Sidebar
st.sidebar.header("Configuration")

# Load Data
@st.cache_data
def load_data():
    output_dir = 'outputs'
    try:
        sentiment_df = pd.read_csv(os.path.join(output_dir, 'sentiment_output.csv'))
        summary_df = pd.read_csv(os.path.join(output_dir, 'per_model_summary.csv'))
        feedback_df = pd.read_csv(os.path.join(output_dir, 'feedback_report.csv'))
        
        absa_path = os.path.join(output_dir, 'absa_training_dataset.csv')
        if os.path.exists(absa_path):
            absa_df = pd.read_csv(absa_path)
        else:
            absa_df = pd.DataFrame()
            
        return sentiment_df, summary_df, feedback_df, absa_df
    except FileNotFoundError:
        return None, None, None, None

sentiment_df, summary_df, feedback_df, absa_df = load_data()

if sentiment_df is None:
    st.error("⚠️ Output files not found! Please run the pipeline first.")
    st.info("Run `python sentiment_pipeline.py` in your terminal.")
    st.stop()

# Sidebar Filters
model_list = ['All'] + sorted(sentiment_df['model'].unique().tolist())
selected_model = st.sidebar.selectbox("Select Model", model_list)

# Filter Data
if selected_model != 'All':
    filtered_df = sentiment_df[sentiment_df['model'] == selected_model]
    filtered_absa = absa_df[absa_df['model_name'] == selected_model] if not absa_df.empty else pd.DataFrame()
    current_summary = summary_df[summary_df['model'] == selected_model]
else:
    filtered_df = sentiment_df
    filtered_absa = absa_df
    current_summary = summary_df

# KPI Metrics
st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

total_reviews = len(filtered_df)
avg_score = filtered_df['sentiment_score'].mean()
pos_pct = (filtered_df['sentiment_label'] == 'Positive').mean() * 100
neg_pct = (filtered_df['sentiment_label'] == 'Negative').mean() * 100

col1.metric("Total Reviews", f"{total_reviews:,}")
col2.metric("Avg Sentiment Score", f"{avg_score:.2f}/5")
col3.metric("Positive Reviews", f"{pos_pct:.1f}%")
col4.metric("Negative Reviews", f"{neg_pct:.1f}%")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualizations", "📝 Feedback Report", "🔍 Raw Data", "🤖 Playground"])

with tab1:
    st.subheader("Sentiment Analysis Visualizations")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Sentiment Distribution
        fig_dist = px.histogram(
            filtered_df, 
            x='sentiment_label', 
            color='sentiment_label',
            category_orders={'sentiment_label': ['Positive', 'Neutral', 'Negative']},
            color_discrete_map={'Positive': '#77DD77', 'Neutral': '#84B6F4', 'Negative': '#FF6961'},
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig_dist, width="stretch")
        
    with c2:
        # Score Distribution
        fig_score = px.histogram(
            filtered_df, 
            x='sentiment_score', 
            nbins=5,
            title="Star Rating Distribution",
            color_discrete_sequence=['#84B6F4']
        )
        fig_score.update_layout(bargap=0.2)
        st.plotly_chart(fig_score, width="stretch")

    if not filtered_absa.empty:
        st.subheader("Aspect-Based Sentiment Analysis (ABSA)")
        
        # Aspect Sentiment Count
        aspect_counts = filtered_absa.groupby(['aspect', 'label']).size().reset_index(name='count')
        
        fig_absa = px.bar(
            aspect_counts, 
            x='aspect', 
            y='count', 
            color='label',
            color_discrete_map={'Positive': '#77DD77', 'Neutral': '#84B6F4', 'Negative': '#FF6961'},
            title="Sentiment by Aspect"
        )
        st.plotly_chart(fig_absa, width="stretch")

with tab2:
    st.subheader("Manufacturer Feedback Report")
    
    if selected_model == 'All':
        st.info("Select a specific model from the sidebar to view its detailed feedback report.")
        st.dataframe(feedback_df)
    else:
        report_row = feedback_df[feedback_df['model'] == selected_model].iloc[0]
        
        st.markdown(f"### Report for **{selected_model}**")
        
        st.info(f"**Summary:** {report_row['summary']}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**✅ Strengths:**\n\n{report_row['strengths']}")
        with c2:
            st.error(f"**⚠️ Weaknesses:**\n\n{report_row['weaknesses']}")
            
        st.warning(f"**💡 Recommendations:**\n\n{report_row['recommendations']}")

with tab3:
    st.subheader("Raw Data Explorer")
    st.dataframe(filtered_df[['model', 'original_review', 'translated_review', 'sentiment_label', 'sentiment_score', 'final_review']])

with tab4:
    st.subheader("Model Playground")
    st.markdown("Test the sentiment model with your own text.")
    
    user_input = st.text_area("Enter a review:", "The battery life is amazing but the camera is terrible.")
    
    if st.button("Analyze"):
        # We need to load the model or use a simple heuristic if model loading is too heavy for the app
        # For this demo, let's try to load the pipeline if available, or mock it
        try:
            from transformers import pipeline
            with st.spinner("Loading model..."):
                sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
                result = sentiment_pipeline(user_input)[0]
                label = result['label']
                score = result['score']
                
                st.write(f"**Prediction:** {label}")
                st.write(f"**Confidence:** {score:.4f}")
                
                # Simple ABSA check
                found_aspects = []
                aspects = ['battery', 'performance', 'display', 'camera', 'build quality', 'price', 'software', 'sound', 'overheating', 'durability']
                for aspect in aspects:
                    if aspect in user_input.lower():
                        found_aspects.append(aspect)
                
                if found_aspects:
                    st.write(f"**Detected Aspects:** {', '.join(found_aspects)}")
                    
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.warning("Ensure transformers is installed and you have internet access.")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit & Python 🐍")
