"""
nlpbert_labeler.py
==================
Separate retraining-specific pipeline module that uses the nlptown BERT model 
(nlptown/bert-base-multilingual-uncased-sentiment) as a teacher to label scraped 
Reddit reviews. This prevents a self-reinforcing bias loop by using a model trained 
on real human star ratings instead of SentimentABSA-v3 itself.
"""

import patch_transformers
import os
import re
import sys
import logging
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from tqdm import tqdm

from aspect_extraction import split_into_sentences, detect_all_aspects
from sentiment_pipeline import preprocess_reviews, translate_and_clean, handle_duplicates

logger = logging.getLogger(__name__)

# Configure logging if run standalone
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_nlptown_classifier():
    """
    Load the nlptown sentiment classifier pipeline once.
    Determines the best available device (CUDA, MPS, or CPU) and fallbacks gracefully if needed.
    """
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    logger.info(f"Loading teacher model: {model_name}...")
    
    device = -1
    if torch.cuda.is_available():
        device = 0
        logger.info("Using CUDA GPU for nlptown teacher model.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS can sometimes have memory or layer compatibility issues, but is faster on Mac
        try:
            device = "mps"
            logger.info("Using Apple Silicon MPS for nlptown teacher model.")
        except Exception as e:
            logger.warning(f"Failed to initialize MPS, falling back to CPU: {e}")
            device = -1
    else:
        logger.info("Using CPU for nlptown teacher model.")

    try:
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            batch_size=64
        )
        # Test classifier to ensure it works on the selected device
        classifier("Test sentence")
        logger.info("✅ Teacher model loaded and verified successfully.")
        return classifier
    except Exception as e:
        logger.warning(f"Error loading model on device {device}: {e}. Falling back to CPU.")
        try:
            classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=-1,
                batch_size=32
            )
            classifier("Test sentence")
            logger.info("✅ Teacher model loaded successfully on CPU fallback.")
            return classifier
        except Exception as fallback_err:
            logger.critical(f"❌ Failed to load nlptown model even on CPU fallback: {fallback_err}")
            raise fallback_err

def map_stars_to_sentiment(star_label: str, score: float) -> tuple[str, float]:
    """
    Map nlptown's star output ("1 star", "2 stars", ..., "5 stars") to standard ABSA labels.
    
    Mapping:
      1-2 stars -> Negative
      3 stars   -> Neutral
      4-5 stars -> Positive
    """
    try:
        # Extract star number (e.g., "5 stars" -> 5 or "1 star" -> 1)
        star_num = int(star_label.split()[0])
    except Exception as e:
        logger.warning(f"Could not parse star rating '{star_label}', defaulting to Neutral: {e}")
        return "Neutral", 0.5

    if star_num >= 4:
        return "Positive", score
    elif star_num <= 2:
        return "Negative", score
    else:
        return "Neutral", score

def label_with_nlptown(df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """
    Run the full retraining data labeling pipeline:
      1. Preprocess reviews (cleaning, meaningless checks, translation, deduplication).
      2. Clause-break reviews into sentences/clauses.
      3. Perform aspect detection per clause, filtering out 'General' aspects.
      4. Label each unique clause using the nlptown model.
      5. Map stars to standard ABSA sentiment labels.
      6. Format into a training DataFrame and deduplicate.
      
    Parameters:
      df: pd.DataFrame with columns: 'model', 'original_review'
      batch_size: batch size for Hugging Face pipeline inference
      
    Returns:
      pd.DataFrame: A DataFrame with columns 'text', 'aspect', 'label', 'confidence', 'model_name'
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to label_with_nlptown.")
        return pd.DataFrame(columns=["text", "aspect", "label", "confidence", "model_name"])

    logger.info(f"Starting NLP-BERT retraining labeling pipeline for {len(df)} reviews...")

    # Step 1: Preprocess, translate & clean, handle duplicates
    proc_df = df[["model", "original_review"]].copy()
    
    # Run the standard data prep pipelines
    proc_df = preprocess_reviews(proc_df)
    proc_df = translate_and_clean(proc_df)
    proc_df = handle_duplicates(proc_df)

    # Filter out invalid reviews
    valid_mask = proc_df["final_review"].notna() & (proc_df["final_review"] != "")
    valid_df = proc_df.loc[valid_mask].copy()

    if valid_df.empty:
        logger.warning("No valid reviews remaining after preprocessing and cleaning.")
        return pd.DataFrame(columns=["text", "aspect", "label", "confidence", "model_name"])

    logger.info(f"Found {len(valid_df)} valid reviews for clause breaking and labeling.")

    # Step 2 & 3: Clause-break and aspect detection
    # Accumulate list of candidate training rows
    candidate_rows = []
    
    for idx, row in valid_df.iterrows():
        text = row["final_review"]
        model_name = row["model"]
        
        # Split into clauses using the canonical spaCy + contrast split
        clauses = split_into_sentences(text)
        
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
                
            # Detect aspects in the clause
            found_aspects = detect_all_aspects(clause)
            
            # Skip clauses with no aspects or only the "General" aspect
            # (the ABSA training dataset strictly requires specific aspects)
            valid_aspects = [asp for asp in found_aspects if asp != "General"]
            
            for aspect in valid_aspects:
                candidate_rows.append({
                    "text": clause,
                    "aspect": aspect,
                    "model_name": model_name
                })

    if not candidate_rows:
        logger.warning("No specific aspects detected in any clauses. No training data generated.")
        return pd.DataFrame(columns=["text", "aspect", "label", "confidence", "model_name"])

    logger.info(f"Generated {len(candidate_rows)} clause-aspect pairs for labeling.")

    # Step 4: Batch label unique clauses to optimize performance
    # Get unique clause texts
    unique_texts = list(set(row["text"] for row in candidate_rows))
    logger.info(f"Labeling {len(unique_texts)} unique clauses using nlptown teacher model...")
    
    classifier = load_nlptown_classifier()
    predictions_map = {}
    
    # Process unique texts in batches
    for i in tqdm(range(0, len(unique_texts), batch_size), desc="Teacher Labeling"):
        batch = unique_texts[i:i + batch_size]
        # Truncate very long texts to 512 characters to avoid model errors
        truncated_batch = [s[:512] if isinstance(s, str) else "" for s in batch]
        
        try:
            results = classifier(truncated_batch, truncation=True, max_length=512)
            for text, res in zip(batch, results):
                label, conf = map_stars_to_sentiment(res["label"], res["score"])
                predictions_map[text] = (label, conf)
        except Exception as e:
            logger.warning(f"Batch prediction failed, falling back to sequential prediction: {e}")
            # Sequential fallback for this batch
            for text, tr_text in zip(batch, truncated_batch):
                try:
                    res = classifier(tr_text, truncation=True, max_length=512)[0]
                    label, conf = map_stars_to_sentiment(res["label"], res["score"])
                    predictions_map[text] = (label, conf)
                except Exception as seq_err:
                    logger.error(f"Failed to predict for text '{text[:40]}...': {seq_err}")
                    predictions_map[text] = ("Neutral", 0.5)

    # Step 5 & 6: Assemble, map, and deduplicate
    final_rows = []
    for item in candidate_rows:
        text = item["text"]
        aspect = item["aspect"]
        model_name = item["model_name"]
        
        label, confidence = predictions_map.get(text, ("Neutral", 0.5))
        
        final_rows.append({
            "text": text,
            "aspect": aspect,
            "label": label,
            "confidence": confidence,
            "model_name": model_name
        })

    result_df = pd.DataFrame(final_rows)
    
    # Deduplicate based on text + aspect to keep a clean ABSA training dataset
    before_dedup = len(result_df)
    result_df = result_df.drop_duplicates(subset=["text", "aspect"])
    logger.info(f"Deduplicated training dataset: {before_dedup} rows -> {len(result_df)} rows.")

    return result_df

if __name__ == "__main__":
    logger.info("Running standalone dry-run of nlpbert_labeler...")
    
    # Create mock DataFrame for testing
    mock_data = pd.DataFrame({
        "model": ["Apple MacBook AIR Apple M2", "Apple MacBook AIR Apple M2", "Apple iPhone 14 Pro"],
        "original_review": [
            "The battery life is absolutely brilliant, easily lasting two whole days! However, the keyboard feels a bit stiff.",
            "I love the gorgeous display and fast performance. It charges very quickly too.",
            "The camera captures stunning photos but it is way too expensive."
        ]
    })
    
    result = label_with_nlptown(mock_data)
    print("\nGenerated ABSA Training Dataset:")
    print("=" * 80)
    print(result.to_string(index=False))
    print("=" * 80)
