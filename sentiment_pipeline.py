import patch_transformers
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

# Unified aspect extraction
from aspect_extraction import (
    split_into_sentences, detect_aspect, detect_all_aspects,
    run_aspect_sentiment, map_sentiment_label,
    ASPECT_KEYWORDS, ASPECT_LIST, SPACY_AVAILABLE,
)

# Constants
FILLER_WORDS = {'lol', 'ok', 'k', 'plz', 'xd'}
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
    # Reduce repeated punctuation BEFORE stripping symbols (e.g. !!! -> !)
    # so sentence boundary markers are preserved
    text = re.sub(r'([!?.])\1+', r'\1', text)
    # Remove emojis/symbols but KEEP ! and ? (needed for sentence splitting)
    text = re.sub(r'[^\w\s,.!?]', '', text) 
    # Remove control characters
    text = re.sub(r'[\n\t\r]', ' ', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
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
            # Use ThreadPoolExecutor to enforce a strict timeout
            import concurrent.futures
            
            def _do_translate():
                return GoogleTranslator(source='auto', target='en').translate(text)
                
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_translate)
                    # 3 second strict timeout per translation
                    translated = future.result(timeout=3)
                return translated
            except concurrent.futures.TimeoutError:
                # If it times out, skip translation to prevent hanging
                return text
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


def analyze_sentiment_absa(df):
    """
    Unified ABSA-first sentiment analysis.

    For every valid review:
      1. Split into sentences / clauses (spaCy + contrast splitting)
      2. Detect aspects in each clause
      3. Run aspect-conditioned sentiment inference per (clause, aspect)
      4. Clauses with no detected aspect → fallback to "General"

    Returns
    -------
    absa_df : pd.DataFrame
        One row per (review, sentence, aspect) with columns:
        review_id, model, original_review, final_review, sentence,
        aspect, sentiment_label, confidence
    classifier : transformers.Pipeline or None
    """
    logger.info("Running ABSA-first sentiment analysis...")

    valid_mask = df['final_review'].notna() & (df['final_review'] != "")
    valid_df = df.loc[valid_mask].copy()

    if valid_df.empty:
        logger.warning("No valid reviews to analyse.")
        return pd.DataFrame(), None

    # ── Load model ──────────────────────────────────────────────────────
    classifier = None
    if TRANSFORMERS_AVAILABLE:
        try:
            model_name = "unknownexplosion/SentimentABSA-v3"
            logger.info(f"Loading Hugging Face model: {model_name}...")
            
            # Programmatically patch tokenizer config to avoid Hugging Face transformers keys bug
            try:
                from huggingface_hub import hf_hub_download
                import json
                config_path = hf_hub_download(repo_id=model_name, filename="tokenizer_config.json")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if isinstance(config.get("extra_special_tokens"), list):
                    config["extra_special_tokens"] = {}
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
            except Exception as patch_err:
                pass

            classifier = pipeline(
                "sentiment-analysis", model=model_name,
                device=-1, model_kwargs={"low_cpu_mem_usage": False},
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    else:
        logger.warning("Transformers not available — using mock sentiment.")

    # ── Process every review ────────────────────────────────────────────
    absa_rows = []

    for review_id, (idx, row) in enumerate(
        tqdm(valid_df.iterrows(), total=len(valid_df), desc="ABSA Analysis")
    ):
        text = row['final_review']
        model_col = row['model']
        original = row.get('original_review', '')

        sentences = split_into_sentences(text)

        review_has_rows = False
        for sent_text in sentences:
            if not sent_text or not sent_text.strip():
                continue

            found_aspects = detect_all_aspects(sent_text)

            # Fallback: if no specific aspect found, use "General"
            if not found_aspects:
                found_aspects = ["General"]

            for aspect in found_aspects:
                if classifier is not None:
                    label, conf = run_aspect_sentiment(
                        sent_text, aspect, classifier
                    )
                else:
                    import random
                    label = random.choice(["Positive", "Negative", "Neutral"])
                    conf = 0.5

                absa_rows.append({
                    'review_id': review_id,
                    'model': model_col,
                    'original_review': original,
                    'final_review': text,
                    'sentence': sent_text,
                    'aspect': aspect,
                    'sentiment_label': label,
                    'confidence': conf,
                })
                review_has_rows = True

        # Safety net: if splitting produced nothing, analyse full review
        if not review_has_rows and text.strip():
            if classifier is not None:
                label, conf = run_aspect_sentiment(text, "General", classifier)
            else:
                label, conf = "Neutral", 0.5
            absa_rows.append({
                'review_id': review_id,
                'model': model_col,
                'original_review': original,
                'final_review': text,
                'sentence': text,
                'aspect': "General",
                'sentiment_label': label,
                'confidence': conf,
            })

    absa_df = pd.DataFrame(absa_rows)
    unique_reviews = absa_df['review_id'].nunique() if not absa_df.empty else 0
    logger.info(
        f"ABSA complete: {unique_reviews} unique reviews → "
        f"{len(absa_df)} aspect-level rows."
    )
    return absa_df, classifier

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

def aggregate_model_stats(absa_df):
    """Aggregates statistics per model from ABSA data.

    Produces two types of rows per model:
      - One row per aspect with aspect-level sentiment breakdown
      - One 'Overall' row using dominant sentiment per unique review
    """
    logger.info("Aggregating statistics per model (ABSA-aware)...")

    if absa_df.empty:
        return pd.DataFrame()

    stats_list = []

    for model in absa_df['model'].unique():
        model_df = absa_df[absa_df['model'] == model]
        unique_reviews = model_df['review_id'].nunique()

        # ── Per-aspect breakdown ────────────────────────────────────────
        for aspect in model_df['aspect'].unique():
            asp_df = model_df[model_df['aspect'] == aspect]
            n = len(asp_df)
            pos = (asp_df['sentiment_label'] == 'Positive').sum()
            neg = (asp_df['sentiment_label'] == 'Negative').sum()
            neu = (asp_df['sentiment_label'] == 'Neutral').sum()
            avg_conf = asp_df['confidence'].mean()

            stats_list.append({
                'model': model,
                'aspect': aspect,
                'total_reviews': unique_reviews,
                'aspect_mentions': n,
                'pct_positive': round(pos / n * 100, 2) if n else 0,
                'pct_negative': round(neg / n * 100, 2) if n else 0,
                'pct_neutral': round(neu / n * 100, 2) if n else 0,
                'avg_confidence': round(avg_conf, 3) if n else 0,
            })

        # ── Overall rollup (dominant sentiment per review) ──────────────
        def _dominant_sentiment(group):
            """Pick the label with the highest cumulative confidence mass."""
            bucket = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}
            for label, conf in zip(group['sentiment_label'], group['confidence']):
                c = float(conf) if pd.notna(conf) else 1.0
                if label in bucket:
                    bucket[label] += c
            return max(bucket, key=bucket.get)

        review_labels = model_df.groupby('review_id').apply(_dominant_sentiment)
        n_overall = len(review_labels)
        pos_o = (review_labels == 'Positive').sum()
        neg_o = (review_labels == 'Negative').sum()
        neu_o = (review_labels == 'Neutral').sum()

        stats_list.append({
            'model': model,
            'aspect': 'Overall',
            'total_reviews': unique_reviews,
            'aspect_mentions': len(model_df),
            'pct_positive': round(pos_o / n_overall * 100, 2) if n_overall else 0,
            'pct_negative': round(neg_o / n_overall * 100, 2) if n_overall else 0,
            'pct_neutral': round(neu_o / n_overall * 100, 2) if n_overall else 0,
            'avg_confidence': round(model_df['confidence'].mean(), 3),
        })

    return pd.DataFrame(stats_list)

def generate_feedback_report(stats_df):
    """Generates the feedback report DataFrame using aspect-level stats."""
    logger.info("Generating structured Manufacturer Feedback Report (ABSA-aware)...")

    if stats_df.empty:
        return pd.DataFrame()

    feedback_list = []

    for model in stats_df['model'].unique():
        model_stats = stats_df[stats_df['model'] == model]
        overall = model_stats[model_stats['aspect'] == 'Overall']
        aspects = model_stats[model_stats['aspect'] != 'Overall']

        if overall.empty:
            continue

        ov = overall.iloc[0]
        pos_pct = ov['pct_positive']
        neg_pct = ov['pct_negative']
        neu_pct = ov['pct_neutral']
        vol = ov['total_reviews']

        # 6.1 Summary
        vol_desc = "High review volume with diverse opinions" if vol > 50 else "Low to moderate volume—insights may be limited"
        sentiment_tone = "overwhelmingly positive" if pos_pct > 75 else "generally positive" if pos_pct > 50 else "mixed" if pos_pct > 30 else "largely negative"

        summary_section = (
            f"6.1 Summary\n"
            f"The overall sentiment for {model} is {sentiment_tone}. "
            f"Analysis shows a breakdown of {pos_pct}% Positive, {neg_pct}% Negative, and {neu_pct}% Neutral sentiment. "
            f"{vol_desc}."
        )

        # 6.2 Strengths — top aspects by positive %
        strong = aspects.sort_values('pct_positive', ascending=False).head(3)
        strength_items = [f"{r['aspect']} ({r['pct_positive']}% positive, {r['aspect_mentions']} mentions)" for _, r in strong.iterrows()]
        strengths_desc = f"Customers praised: {'; '.join(strength_items)}." if not strong.empty else "No specific praise themes detected."

        strengths_section = f"6.2 Strengths\n{strengths_desc}"

        # 6.3 Weaknesses — top aspects by negative %
        weak = aspects[aspects['pct_negative'] > 10].sort_values('pct_negative', ascending=False).head(3)
        if not weak.empty:
            weak_items = [f"{r['aspect']} ({r['pct_negative']}% negative, {r['aspect_mentions']} mentions)" for _, r in weak.iterrows()]
            weaknesses_desc = f"recurring complaints focused on: {'; '.join(weak_items)}."
        else:
            weaknesses_desc = "No significant complaint themes detected."

        weaknesses_section = f"6.3 Weaknesses\nCritical areas for improvement include {weaknesses_desc}"

        # 6.4 Actionable Recommendations (aspect-driven)
        recs = []
        weak_aspects = set(weak['aspect'].tolist()) if not weak.empty else set()
        if 'Battery' in weak_aspects:
            recs.append("Technical: Optimize background process management to extend battery life.")
        if 'Display' in weak_aspects:
            recs.append("Quality Control: Investigate display panel durability and consider stronger protective glass.")
        if 'Price' in weak_aspects:
            recs.append("Marketing: Emphasize long-term value and premium build quality to justify pricing.")
        if 'Camera' in weak_aspects:
            recs.append("Product: Refine image processing algorithms for better low-light performance.")
        if 'Heating / Thermals' in weak_aspects:
            recs.append("Engineering: Improve thermal management to reduce overheating during intensive tasks.")
        if 'Software & OS' in weak_aspects:
            recs.append("Software: Prioritize bug fixes and stability improvements in upcoming OS updates.")

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
        recommendations_section = f"6.4 Actionable Recommendations\n{rec_list_str}"

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

def plot_results(absa_df, stats_df, output_dir):
    """Generates and saves ABSA-aware plots."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving plots to {output_dir}...")

    if absa_df.empty:
        logger.warning("No data to plot.")
        return

    # 1. Global Sentiment Distribution (dominant sentiment per review)
    def _dominant_sentiment_global(group):
        """Pick the label with the highest cumulative confidence mass."""
        bucket = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}
        for label, conf in zip(group['sentiment_label'], group['confidence']):
            c = float(conf) if pd.notna(conf) else 1.0
            if label in bucket:
                bucket[label] += c
        return max(bucket, key=bucket.get)

    review_sentiments = absa_df.groupby('review_id').apply(_dominant_sentiment_global).reset_index()
    review_sentiments.columns = ['review_id', 'sentiment_label']

    plt.figure(figsize=(8, 6))
    sns.countplot(data=review_sentiments, x='sentiment_label',
                  order=['Positive', 'Neutral', 'Negative'])
    plt.title('Global Sentiment Distribution (Review-Level)')
    plt.savefig(os.path.join(output_dir, 'global_sentiment_distribution.png'))
    plt.close()

    # 2. Aspect Sentiment Heatmap (stacked bar)
    aspect_counts = absa_df[absa_df['aspect'] != 'General']['aspect'].value_counts().head(10)
    top_aspects = aspect_counts.index.tolist()
    if top_aspects:
        asp_data = absa_df[absa_df['aspect'].isin(top_aspects)]
        ct = pd.crosstab(asp_data['aspect'], asp_data['sentiment_label'], normalize='index') * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        ct_ordered = ct.reindex(columns=['Positive', 'Neutral', 'Negative'], fill_value=0)
        ct_ordered.plot(kind='barh', stacked=True, ax=ax,
                        color=['#34C759', '#8E8E93', '#FF3B30'])
        ax.set_xlabel('Percentage %')
        ax.set_title('Sentiment per Aspect (Top 10)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aspect_sentiment_heatmap.png'))
        plt.close()

    # 3. Per-model Sentiment (dominant per review)
    review_models = absa_df.drop_duplicates('review_id')[['review_id', 'model']]
    review_merged = review_sentiments.merge(review_models, on='review_id')

    plt.figure(figsize=(12, 6))
    sns.countplot(data=review_merged, x='model', hue='sentiment_label',
                  hue_order=['Positive', 'Neutral', 'Negative'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Sentiment Count per Model (Review-Level)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_model_sentiment_count.png'))
    plt.close()


def main():
    dataset_path = 'final_dataset.csv'
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

    # 5. ABSA-first Sentiment Analysis (single unified pass)
    absa_df, classifier = analyze_sentiment_absa(df)

    # 6. Aggregate
    stats_df = aggregate_model_stats(absa_df)

    # 7. Feedback
    feedback_df = generate_feedback_report(stats_df)

    # 8. Save Outputs
    logger.info("Saving outputs...")
    absa_df.to_csv(os.path.join(output_dir, 'sentiment_output.csv'), index=False)
    stats_df.to_csv(os.path.join(output_dir, 'per_model_summary.csv'), index=False)
    feedback_df.to_csv(os.path.join(output_dir, 'feedback_report.csv'), index=False)

    # Backward-compatible training dataset (aspect != "General")
    training_df = absa_df[absa_df['aspect'] != 'General'].copy()
    if not training_df.empty:
        # Rename columns to match expected training format
        train_export = training_df.rename(columns={
            'sentence': 'text',
            'sentiment_label': 'label',
            'model': 'model_name',
        })[['text', 'aspect', 'label', 'confidence', 'model_name']]
        train_export.to_csv(os.path.join(output_dir, 'absa_training_dataset.csv'), index=False)

    # Markdown report
    md_path = os.path.join(output_dir, 'manufacturer_recommendations.md')
    with open(md_path, 'w') as f:
        f.write("# Manufacturer Feedback Report\n\n")
        for _, row in feedback_df.iterrows():
            f.write(f"## Model: {row['model']}\n")
            f.write(row['full_report_text'])
            f.write("\n\n")
            f.write("---\n\n")

    # 9. Plots
    try:
        plot_results(absa_df, stats_df, plots_dir)
    except ImportError:
        logger.warning("Matplotlib/Seaborn not found. Skipping plots.")
    except Exception as e:
        logger.warning(f"Error plotting: {e}")

    # 10. Display
    print("\n=== Model Summary (First 10 Rows) ===")
    try:
        print(stats_df.head(10).to_markdown(index=False))
    except ImportError:
        print(stats_df.head(10).to_string(index=False))

    print("\n=== Feedback Report (First 5 Models) ===")
    try:
        print(feedback_df.head(5).to_markdown(index=False))
    except ImportError:
        print(feedback_df.head(5).to_string(index=False))

    unique = absa_df['review_id'].nunique() if not absa_df.empty else 0
    print(f"\nPipeline completed. {unique} reviews → {len(absa_df)} aspect rows. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
