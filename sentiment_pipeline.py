import pandas as pd
import numpy as np
import re
import os
import sys
import logging
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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

# Check for Spacy
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading en_core_web_sm...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("Spacy not found. ABSA extraction will be skipped.")
    SPACY_AVAILABLE = False

# Constants
FILLER_WORDS = {'lol', 'ok', 'k', 'plz', 'xd'}
MIN_ALPHA_CHARS = 3
MIN_ALPHANUM_RATIO = 0.3
ASPECT_LIST = ['battery', 'performance', 'display', 'camera', 'build quality', 'price', 'software', 'sound', 'overheating', 'durability']

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
    tqdm.pandas(desc="Translating & Cleaning")
    df['translated_review'] = df['cleaned_review'].progress_apply(lambda x: process_row({'cleaned_review': x}))
    
    # Re-clean translated text
    df['translated_review'] = df['translated_review'].apply(clean_text)
    
    # Re-check meaningless
    mask_meaningless = df['translated_review'].apply(is_meaningless)
    df.loc[mask_meaningless, 'translated_review'] = np.nan
    
    # Use translated review as the final 'cleaned_review' for sentiment analysis
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
        return df, None

    scores = []
    labels = []
    sentiment_pipeline = None

    if TRANSFORMERS_AVAILABLE:
        try:
            local_model_path = "outputs/fine_tuned_absa_model"
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            
            if os.path.exists(local_model_path):
                logger.info(f"Found fine-tuned model at {local_model_path}. Using it!")
                model_name = local_model_path
            else:
                logger.info(f"Fine-tuned model not found. Using default: {model_name}")

            logger.info(f"Loading model: {model_name}...")
            sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
            
            # Process in batches to avoid memory issues
            batch_size = 32
            logger.info(f"Processing {len(valid_reviews)} reviews in batches of {batch_size}...")
            
            for i in tqdm(range(0, len(valid_reviews), batch_size), desc="Sentiment Analysis"):
                batch = valid_reviews[i:i+batch_size]
                # Truncate to 512 tokens to avoid errors
                results = sentiment_pipeline(batch, truncation=True, max_length=512)
                
                for res in results:
                    label_str = res['label']
                    
                    # Handle "N stars" format (BERT base)
                    if label_str[0].isdigit():
                        star = int(label_str.split()[0])
                        scores.append(star)
                        labels.append(get_sentiment_label(star))
                    
                    # Handle "Positive/Negative" format (Fine-Tuned DeBERTa)
                    else:
                        # Map text labels to approximate star scores for compatibility
                        lower_label = label_str.lower()
                        if 'positive' in lower_label:
                            scores.append(5)
                            labels.append('Positive')
                        elif 'negative' in lower_label:
                            scores.append(1)
                            labels.append('Negative')
                        else:
                            scores.append(3)
                            labels.append('Neutral')
                    
        except Exception as e:
            logger.error(f"Error in transformers pipeline: {e}")
            # Fallback to mock
            scores = [3] * len(valid_reviews)
            labels = ['Neutral'] * len(valid_reviews)
    else:
        logger.warning("Using MOCK sentiment analysis (Random) because transformers is missing.")
        import random
        for _ in valid_reviews:
            s = random.randint(1, 5)
            scores.append(s)
            labels.append(get_sentiment_label(s))

    # Assign back to DataFrame
    df.loc[valid_mask, 'sentiment_score'] = scores
    df.loc[valid_mask, 'sentiment_label'] = labels
    
    return df, sentiment_pipeline

def generate_absa_dataset(df, sentiment_pipeline):
    """Generates ABSA dataset for DeBERTa fine-tuning."""
    if not SPACY_AVAILABLE:
        logger.warning("Spacy not available, skipping ABSA dataset generation.")
        return pd.DataFrame()

    logger.info("Generating ABSA dataset...")
    absa_data = []
    
    # Iterate over valid reviews
    valid_df = df[df['final_review'].notna() & (df['final_review'] != "")]
    
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="ABSA Extraction"):
        text = row['final_review']
        model_name = row['model']
        
        doc = nlp(text)
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Check for aspects
            found_aspects = []
            for aspect in ASPECT_LIST:
                # Simple keyword matching (can be enhanced with synonyms)
                if aspect in sent_text.lower():
                    found_aspects.append(aspect)
            
            if found_aspects:
                # Determine sentiment of the sentence
                # If we have the pipeline, use it on the sentence
                # Otherwise fall back to review sentiment (less accurate)
                label = 'Neutral'
                if sentiment_pipeline:
                    try:
                        res = sentiment_pipeline(sent_text, truncation=True, max_length=512)[0]
                        star = int(res['label'].split()[0])
                        label = get_sentiment_label(star)
                    except:
                        label = row['sentiment_label'] if pd.notna(row['sentiment_label']) else 'Neutral'
                else:
                    label = row['sentiment_label'] if pd.notna(row['sentiment_label']) else 'Neutral'
                
                for aspect in found_aspects:
                    absa_data.append({
                        'text': sent_text,
                        'aspect': aspect,
                        'label': label,
                        'model_name': model_name
                    })
                    
    absa_df = pd.DataFrame(absa_data)
    logger.info(f"Generated {len(absa_df)} ABSA training samples.")
    return absa_df

def extract_keywords(text_series, top_n=6):
    """Extracts top keywords excluding common stop words."""
    if text_series.empty:
        return []
    
    all_text = " ".join(text_series.astype(str).tolist()).lower()
    # Remove punctuation
    all_text = re.sub(r'[^\w\s]', '', all_text)
    words = all_text.split()
    
    # Basic stop words list
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
    """Generates the feedback report DataFrame strictly following the 6.1-6.4 format."""
    logger.info("Generating structured Manufacturer Feedback Report...")
    
    feedback_list = []
    
    for _, row in stats_df.iterrows():
        model = row['model']
        pos_pct = row['pct_positive']
        neg_pct = row['pct_negative']
        neu_pct = row['pct_neutral']
        vol = row['valid_cleaned_reviews']
        
        # 6.1 Summary
        vol_desc = "High review volume with diverse opinions" if vol > 50 else "Low to moderate volume—insights may be limited"
        sentiment_tone = "overwhelmingly positive" if pos_pct > 75 else "generally positive" if pos_pct > 50 else "mixed" if pos_pct > 30 else "largely negative"
        
        summary_section = (
            f"6.1 Summary\n"
            f"The overall sentiment for {model} is {sentiment_tone}. "
            f"Analysis shows a breakdown of {pos_pct}% Positive, {neg_pct}% Negative, and {neu_pct}% Neutral sentiment. "
            f"{vol_desc}."
        )
        
        # 6.2 Strengths
        pos_kw = row['top_6_positive_keywords']
        strengths_desc = f"Customers frequently praised key aspects such as: {pos_kw}." if pos_kw else "No specific praise themes detected."
        
        strengths_section = (
            f"6.2 Strengths\n"
            f"{strengths_desc}"
        )
        
        # 6.3 Weaknesses
        neg_kw = row['top_6_negative_keywords']
        weaknesses_desc = f"recurring complaints focused on: {neg_kw}." if neg_kw else "No specific complaint themes detected."
        
        weaknesses_section = (
            f"6.3 Weaknesses\n"
            f"Critical areas for improvement include {weaknesses_desc}"
        )
        
        # 6.4 Actionable Recommendations
        recs = []
        
        # Dynamic advice based on keywords in weaknesses
        w_text = str(neg_kw).lower()
        if "battery" in w_text or "drain" in w_text:
            recs.append("Technical: Optimize background process management to extend battery life.")
        if "screen" in w_text or "display" in w_text or "crack" in w_text:
            recs.append("Quality Control: Investigate display panel durability and consider stronger protective glass.")
        if "price" in w_text or "cost" in w_text or "expensive" in w_text:
            recs.append("Marketing: Emphasize long-term value and premium build quality to justify pricing.")
        if "camera" in w_text or "photo" in w_text:
            recs.append("Product: Refine image processing algorithms for better low-light performance.")
        if "delivery" in w_text or "shipping" in w_text:
            recs.append("Logistics: Review courier partners to reduce shipping delays and damage.")
        if "service" in w_text or "support" in w_text:
            recs.append("Trust: Expand customer support availability and reduce response times.")
            
        # Fillers if we don't have enough specific ones
        while len(recs) < 3:
            defaults = [
                "Marketing: Highlight the most praised features (Strengths) in upcoming ad campaigns.",
                "Engagement: packaging enhancements to create a more premium unboxing experience.",
                "Trust: Respond publicly to constructive negative reviews to show brand accountability."
            ]
            for d in defaults:
                if d not in recs:
                    recs.append(d)
                    if len(recs) >= 3: break
        
        rec_list_str = "\n".join([f"- {r}" for r in recs])
        recommendations_section = (
            f"6.4 Actionable Recommendations\n"
            f"{rec_list_str}"
        )
        
        # Combine full text for the CSV report
        full_report = f"{summary_section}\n\n{strengths_section}\n\n{weaknesses_section}\n\n{recommendations_section}"
        
        feedback_list.append({
            'model': model,
            'summary': summary_section, 
            'strengths': strengths_section,
            'weaknesses': weaknesses_section,
            'recommendations': recommendations_section,
            'full_report_text': full_report
        })
        
    return pd.DataFrame(feedback_list)

def plot_results(df, stats_df, output_dir):
    """Generates and saves plots."""
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
    
    # Top Keywords (Horizontal Bar) - Just taking top 10 overall for simplicity
    plt.figure(figsize=(10, 8))
    all_valid_reviews = df[df['sentiment_label'].notna()]['final_review']
    top_keywords = extract_keywords(all_valid_reviews, top_n=10)
    if top_keywords:
        # We need counts for plotting, so let's re-do a bit of logic here or just mock it for visual
        # Better: Re-use the counter logic
        all_text = " ".join(all_valid_reviews.astype(str).tolist()).lower()
        all_text = re.sub(r'[^\w\s]', '', all_text)
        words = all_text.split()
        stop_words = {'the', 'and', 'a', 'to', 'of', 'in', 'it', 'is', 'i', 'for', 'that', 'you', 'my', 'with', 'on', 'this', 'was', 'but', 'so', 'have', 'be', 'not', 'are', 'as', 'at', 'if', 'or', 'me', 'one', 'up', 'out', 'all', 'very', 'good', 'great', 'product', 'laptop', 'phone', 'device', 'its', 'just', 'like', 'from', 'an', 'no', 'has', 'had', 'will', 'can', 'do', 'about', 'when', 'get', 'use', 'than', 'more', 'some', 'only', 'would', 'really', 'after', 'time', 'buy', 'best', 'well', 'much', 'also', 'even', 'too', 'am', 'because', 'don', 't', 's', 've', 'm', 're', 'd', 'll'}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        counter = Counter(filtered_words)
        common = counter.most_common(10)
        
        y = [x[0] for x in common]
        x = [x[1] for x in common]
        
        sns.barplot(x=x, y=y)
        plt.title('Top 10 Keywords Overall')
        plt.xlabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_keywords.png'))
        plt.close()

def main():
    dataset_path = '/Users/anubhavmukherjee/Desktop/Sentiment-analysis/final_dataset.csv'
    output_dir = '/Users/anubhavmukherjee/Desktop/Sentiment-analysis/outputs'
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load
    df = load_data(dataset_path)
    # df = df.head(100) # Uncomment for testing
    
    # 2. Preprocess
    df = preprocess_reviews(df)
    
    # 3. Translate
    df = translate_and_clean(df)
    
    # 4. Handle Duplicates
    df = handle_duplicates(df)
    
    # 5. Sentiment Analysis
    df, sentiment_pipeline = analyze_sentiment(df)
    
    # 6. ABSA Dataset Generation
    absa_df = generate_absa_dataset(df, sentiment_pipeline)
    
    # 7. Aggregate
    stats_df = aggregate_model_stats(df)
    
    # 8. Feedback
    feedback_df = generate_feedback_report(stats_df)
    
    # 9. Save Outputs
    logger.info("Saving outputs...")
    df.to_csv(os.path.join(output_dir, 'sentiment_output.csv'), index=False)
    stats_df.to_csv(os.path.join(output_dir, 'per_model_summary.csv'), index=False)
    feedback_df.to_csv(os.path.join(output_dir, 'feedback_report.csv'), index=False)
    if not absa_df.empty:
        absa_df.to_csv(os.path.join(output_dir, 'absa_training_dataset.csv'), index=False)
    
    # Markdown report
    md_path = os.path.join(output_dir, 'manufacturer_recommendations.md')
    with open(md_path, 'w') as f:
        f.write("# Manufacturer Feedback Report\n\n")
        for _, row in feedback_df.iterrows():
            f.write(f"## Model: {row['model']}\n")
            f.write(row['full_report_text'])
            f.write("\n\n")
            f.write("---\n\n")
            
    # 10. Plots
    try:
        plot_results(df, stats_df, plots_dir)
    except ImportError:
        logger.warning("Matplotlib/Seaborn not found. Skipping plots.")
    except Exception as e:
        logger.warning(f"Error plotting: {e}")
    
    # 11. Display
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
