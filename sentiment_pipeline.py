import pandas as pd
import numpy as np
import re
import os
import sys
import logging
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for ML libraries
try:
    from langdetect import detect
    from deep_translator import GoogleTranslator
    TRANS_AVAILABLE = True
except ImportError:
    logger.warning("Translation libraries not found. Translation will be skipped.")
    TRANS_AVAILABLE = False

# Check for Transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not found. Sentiment analysis will use mock/fallback.")
    TRANSFORMERS_AVAILABLE = False

# Constants
FILLER_WORDS = {'lol', 'xd', 'lmao', 'plz', 'ok', 'k'}
MIN_ALPHA_CHARS = 3
MIN_ALPHANUM_RATIO = 0.3

def load_data(filepath):
    """Loads the dataset and keeps only relevant columns."""
    logger.info(f"Loading data from {filepath}...")
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format")
        
        # Keep first two columns and rename
        df = df.iloc[:, :2]
        df.columns = ['model', 'original_review']
        logger.info(f"Loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def clean_text(text):
    """Applies cleaning rules to the review text."""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove emojis (simple range based, can be improved with emoji lib but regex is faster/lighter)
    text = re.sub(r'[^\w\s,.]', '', text) 
    # Remove control characters
    text = re.sub(r'[\n\t\r]', ' ', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Reduce repeated punctuation (e.g. !! -> !)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    
    return text

def is_meaningless(text):
    """Checks if the review is meaningless based on heuristics."""
    if not text or not isinstance(text, str):
        return True
    
    # Check for empty/whitespace
    if not text.strip():
        return True
    
    # Check for filler words
    words = set(text.lower().split())
    if words.issubset(FILLER_WORDS):
        return True
    
    # Check for alphabetic characters count
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count < MIN_ALPHA_CHARS:
        return True
    
    # Check for alphanumeric ratio
    alphanum_count = sum(c.isalnum() for c in text)
    if len(text) > 0 and (alphanum_count / len(text)) < MIN_ALPHANUM_RATIO:
        return True
        
    return False

def preprocess_reviews(df):
    """Applies cleaning and meaningless checks."""
    logger.info("Cleaning reviews...")
    
    # Initial cleaning
    df['cleaned_review'] = df['original_review'].apply(clean_text)
    
    # Identify meaningless reviews
    mask_meaningless = df['cleaned_review'].apply(is_meaningless)
    df.loc[mask_meaningless, 'cleaned_review'] = np.nan
    
    logger.info(f"Found {mask_meaningless.sum()} meaningless reviews after initial cleaning.")
    return df

def translate_and_clean(df):
    """Detects language, translates to English if needed, and re-cleans."""
    if not TRANS_AVAILABLE:
        logger.warning("Skipping translation step as libraries are missing.")
        df['translated_review'] = df['cleaned_review']
        df['final_review'] = df['cleaned_review']
        return df

    logger.info("Starting language detection and translation...")
    
    def process_row(row):
        text = row['cleaned_review']
        if pd.isna(text) or text == "":
            return text
            
        try:
            lang = detect(text)
        except:
            lang = 'unknown'
            
        if lang != 'en' and lang != 'unknown':
            try:
                # Translate
                translated = GoogleTranslator(source='auto', target='en').translate(text)
                return translated
            except Exception as e:
                return text # Fallback to original
        return text

    # Apply translation only to non-NaN rows
    # Note: This can be slow for large datasets.
    # For optimization, we could filter first.
    df['translated_review'] = df['cleaned_review'].apply(lambda x: process_row({'cleaned_review': x}))
    
    # Re-clean translated text
    df['translated_review'] = df['translated_review'].apply(clean_text)
    
    # Re-check meaningless
    mask_meaningless = df['translated_review'].apply(is_meaningless)
    df.loc[mask_meaningless, 'translated_review'] = np.nan
    
    # Use translated review as the final 'cleaned_review' for sentiment analysis
    # If translation failed or wasn't needed, it holds the original cleaned text
    df['final_review'] = df['translated_review']
    
    return df

def handle_duplicates(df):
    """Handles duplicates per model. Keeps first, marks others as NaN."""
    logger.info("Handling duplicates...")
    
    # Create a normalized column for comparison
    df['norm_review'] = df['final_review'].astype(str).str.lower().str.strip()
    
    # Identify duplicates per model
    # keep='first' marks duplicates as True (except the first occurrence)
    duplicates = df.duplicated(subset=['model', 'norm_review'], keep='first')
    
    # Set final_review to NaN for duplicates
    df.loc[duplicates, 'final_review'] = np.nan
    
    logger.info(f"Marked {duplicates.sum()} duplicate reviews as NaN.")
    return df

def get_sentiment_label(score):
    """Maps 1-5 stars to Positive, Negative, Neutral."""
    # 1, 2 -> Negative
    # 3 -> Neutral
    # 4, 5 -> Positive
    if score <= 2:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'

def analyze_sentiment(df):
    """Runs sentiment analysis on valid reviews."""
    logger.info("Running sentiment analysis...")
    
    # Filter for valid rows
    valid_mask = df['final_review'].notna() & (df['final_review'] != "")
    valid_reviews = df.loc[valid_mask, 'final_review'].tolist()
    
    if not valid_reviews:
        logger.warning("No valid reviews to analyze.")
        df['sentiment_score'] = np.nan
        df['sentiment_label'] = np.nan
        return df

    scores = []
    labels = []

    if TRANSFORMERS_AVAILABLE:
        try:
            logger.info("Loading BERT model...")
            sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
            
            # Process in batches to avoid memory issues
            batch_size = 32
            logger.info(f"Processing {len(valid_reviews)} reviews in batches of {batch_size}...")
            
            for i in range(0, len(valid_reviews), batch_size):
                batch = valid_reviews[i:i+batch_size]
                # Truncate to 512 tokens to avoid errors
                results = sentiment_pipeline(batch, truncation=True, max_length=512)
                
                for res in results:
                    # Label is like '1 star', '5 stars'
                    star = int(res['label'].split()[0])
                    scores.append(star)
                    labels.append(get_sentiment_label(star))
                    
        except Exception as e:
            logger.error(f"Error in transformers pipeline: {e}")
            # Fallback to mock
            scores = [3] * len(valid_reviews)
            labels = ['Neutral'] * len(valid_reviews)
    else:
        logger.warning("Using MOCK sentiment analysis (Random) because transformers is missing.")
        # Mock logic for testing
        import random
        for _ in valid_reviews:
            s = random.randint(1, 5)
            scores.append(s)
            labels.append(get_sentiment_label(s))

    # Assign back to DataFrame
    df.loc[valid_mask, 'sentiment_score'] = scores
    df.loc[valid_mask, 'sentiment_label'] = labels
    
    return df

def extract_keywords(text_series, top_n=6):
    """Extracts top keywords excluding common stop words."""
    if text_series.empty:
        return []
    
    all_text = " ".join(text_series.astype(str).tolist()).lower()
    # Remove punctuation
    all_text = re.sub(r'[^\w\s]', '', all_text)
    words = all_text.split()
    
    # Basic stop words list (can be expanded)
    stop_words = {'the', 'and', 'a', 'to', 'of', 'in', 'it', 'is', 'i', 'for', 'that', 'you', 'my', 'with', 'on', 'this', 'was', 'but', 'so', 'have', 'be', 'not', 'are', 'as', 'at', 'if', 'or', 'me', 'one', 'up', 'out', 'all', 'very', 'good', 'great', 'product', 'laptop', 'phone', 'device', 'its', 'just', 'like', 'from', 'an', 'no', 'has', 'had', 'will', 'can', 'do', 'about', 'when', 'get', 'use', 'than', 'more', 'some', 'only', 'would', 'really', 'after', 'time', 'buy', 'best', 'well', 'much', 'also', 'even', 'too', 'am', 'because', 'don', 't', 's', 've', 'm', 're', 'd', 'll'}
    
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    counter = Counter(filtered_words)
    return [word for word, count in counter.most_common(top_n)]

def aggregate_model_stats(df):
    """Aggregates statistics per model."""
    logger.info("Aggregating statistics per model...")
    
    models = df['model'].unique()
    stats_list = []
    
    for model in models:
        model_df = df[df['model'] == model]
        total_reviews = len(model_df)
        
        # Valid reviews are those with a sentiment score
        valid_df = model_df[model_df['sentiment_label'].notna()]
        valid_count = len(valid_df)
        
        if valid_count > 0:
            pos_count = (valid_df['sentiment_label'] == 'Positive').sum()
            neg_count = (valid_df['sentiment_label'] == 'Negative').sum()
            neu_count = (valid_df['sentiment_label'] == 'Neutral').sum()
            
            avg_score = valid_df['sentiment_score'].mean()
            
            pct_pos = (pos_count / valid_count) * 100
            pct_neg = (neg_count / valid_count) * 100
            pct_neu = (neu_count / valid_count) * 100
            
            # Keywords
            pos_reviews = valid_df[valid_df['sentiment_label'] == 'Positive']['final_review']
            neg_reviews = valid_df[valid_df['sentiment_label'] == 'Negative']['final_review']
            
            top_pos_kw = extract_keywords(pos_reviews)
            top_neg_kw = extract_keywords(neg_reviews)
            
        else:
            pct_pos = pct_neg = pct_neu = avg_score = 0
            top_pos_kw = []
            top_neg_kw = []
        
        stats_list.append({
            'model': model,
            'total_reviews': total_reviews,
            'valid_cleaned_reviews': valid_count,
            'pct_positive': round(pct_pos, 2),
            'pct_negative': round(pct_neg, 2),
            'pct_neutral': round(pct_neu, 2),
            'average_sentiment_score': round(avg_score, 2),
            'top_6_positive_keywords': ", ".join(top_pos_kw),
            'top_6_negative_keywords': ", ".join(top_neg_kw)
        })
        
    return pd.DataFrame(stats_list)

def generate_feedback_report(stats_df):
    """Generates the feedback report DataFrame."""
    logger.info("Generating feedback report...")
    
    feedback_list = []
    
    for _, row in stats_df.iterrows():
        model = row['model']
        
        # Summary
        sentiment_desc = "generally positive" if row['pct_positive'] > row['pct_negative'] else "mixed or negative"
        summary = (f"The sentiment for {model} is {sentiment_desc} with {row['pct_positive']}% positive reviews. "
                   f"Based on {row['valid_cleaned_reviews']} valid reviews, the average sentiment score is {row['average_sentiment_score']}/5.")
        
        # Strengths & Weaknesses
        strengths = row['top_6_positive_keywords'] if row['top_6_positive_keywords'] else "No clear strengths identified."
        weaknesses = row['top_6_negative_keywords'] if row['top_6_negative_keywords'] else "No clear weaknesses identified."
        
        # Recommendations (Template based)
        recs = []
        if row['pct_negative'] > 20:
            recs.append("Address common complaints found in negative reviews immediately.")
        if "battery" in weaknesses:
            recs.append("Investigate battery life optimization or quality control.")
        if "screen" in weaknesses or "display" in weaknesses:
            recs.append("Review display panel quality and durability.")
        if "price" in weaknesses:
            recs.append("Consider reviewing pricing strategy or adding more value.")
        if "service" in weaknesses:
            recs.append("Improve customer service and support channels.")
            
        # Generic fallback recommendations
        if len(recs) < 3:
            recs.append("Engage with customers on social media to build trust.")
            recs.append("Highlight top-rated features in marketing campaigns.")
            recs.append("Incentivize happy customers to leave detailed reviews.")
            
        recommendations = "; ".join(recs[:3]) # Take top 3
        
        feedback_list.append({
            'model': model,
            'summary': summary,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations
        })
        
    return pd.DataFrame(feedback_list)

def plot_results(df, stats_df, output_dir):
    """Generates and saves plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving plots to {output_dir}...")
    
    # Global Sentiment Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sentiment_label', order=['Positive', 'Neutral', 'Negative'])
    plt.title('Global Sentiment Distribution')
    plt.savefig(os.path.join(output_dir, 'global_sentiment_distribution.png'))
    plt.close()
    
    # Per-model Sentiment Count
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='model', hue='sentiment_label', hue_order=['Positive', 'Neutral', 'Negative'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Sentiment Count per Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_model_sentiment_count.png'))
    plt.close()

def main():
    dataset_path = '/Users/anubhavmukherjee/Desktop/Sentiment-analysis/final_dataset.csv'
    output_dir = '/Users/anubhavmukherjee/Desktop/Sentiment-analysis/outputs'
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load
    df = load_data(dataset_path)
    
    # 2. Preprocess
    df = preprocess_reviews(df)
    
    # 3. Translate
    df = translate_and_clean(df)
    
    # 4. Handle Duplicates
    df = handle_duplicates(df)
    
    # 5. Sentiment Analysis
    df = analyze_sentiment(df)
    
    # 6. Aggregate
    stats_df = aggregate_model_stats(df)
    
    # 7. Feedback
    feedback_df = generate_feedback_report(stats_df)
    
    # 8. Save Outputs
    logger.info("Saving outputs...")
    df.to_csv(os.path.join(output_dir, 'sentiment_output.csv'), index=False)
    stats_df.to_csv(os.path.join(output_dir, 'per_model_summary.csv'), index=False)
    feedback_df.to_csv(os.path.join(output_dir, 'feedback_report.csv'), index=False)
    
    # Markdown report
    md_path = os.path.join(output_dir, 'manufacturer_recommendations.md')
    with open(md_path, 'w') as f:
        f.write("# Manufacturer Feedback Report\\n\\n")
        for _, row in feedback_df.iterrows():
            f.write(f"## Model: {row['model']}\\n")
            f.write(f"**Summary**: {row['summary']}\\n\\n")
            f.write(f"**Strengths**: {row['strengths']}\\n\\n")
            f.write(f"**Weaknesses**: {row['weaknesses']}\\n\\n")
            f.write(f"**Recommendations**: {row['recommendations']}\\n\\n")
            f.write("---\\n\\n")
            
    # 9. Plots
    try:
        plot_results(df, stats_df, plots_dir)
    except ImportError:
        logger.warning("Matplotlib/Seaborn not found. Skipping plots.")
    except Exception as e:
        logger.warning(f"Error plotting: {e}")
    
    # 10. Display (Print to console for now, Notebook will handle display)
    print("\n=== Model Summary (First 10 Rows) ===")
    try:
        print(stats_df.head(10).to_markdown(index=False))
    except ImportError:
        print(stats_df.head(10).to_string(index=False))
    
    print("\n=== Feedback Report (First 10 Rows) ===")
    try:
        print(feedback_df.head(10).to_markdown(index=False))
    except ImportError:
        print(feedback_df.head(10).to_string(index=False))
    
    print(f"\nPipeline completed. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
