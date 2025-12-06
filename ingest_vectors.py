import os
import json
import logging
import time
import pandas as pd
import google.generativeai as genai
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
CSV_PATH = "outputs/sentiment_output.csv"
DB_NAME = "sentiment_analysis_db"
COLLECTION_NAME = "review_embeddings"
EMBEDDING_MODEL = "models/text-embedding-004"
BATCH_SIZE = 50  # Batch size for embedding generation

# Load Secrets
def load_secrets():
    secrets = {}
    try:
        if os.path.exists(".streamlit/secrets.toml"):
            with open(".streamlit/secrets.toml", "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.split("=", 1)
                        secrets[key.strip()] = value.strip().replace('"', '')
    except Exception as e:
        logger.error(f"Error loading secrets: {e}")
    return secrets

secrets = load_secrets()
MONGO_URI = secrets.get("MONGO_URI")
GOOGLE_API_KEY = secrets.get("GOOGLE_API_KEY")

if not MONGO_URI or not GOOGLE_API_KEY:
    logger.error("❌ Missing Secrets. Please check .streamlit/secrets.toml")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

def ingest_data():
    # 1. Connect to MongoDB
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        logger.info("✅ Connected to MongoDB.")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return

    # 2. Load Data
    if not os.path.exists(CSV_PATH):
        logger.error(f"❌ File not found: {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    total_reviews = len(df)
    logger.info(f"📊 Loaded {total_reviews} reviews from CSV.")

    # 3. Batch Processing
    operations = []
    
    # We will process in batches to be efficient
    for i in tqdm(range(0, total_reviews, BATCH_SIZE), desc="Generating Embeddings"):
        batch = df.iloc[i : i + BATCH_SIZE]
        
        # Robust filtering: ensure string and non-empty
        valid_batch = batch[batch['final_review'].astype(str).str.strip().astype(bool)]
        
        if valid_batch.empty:
            continue

        texts = valid_batch['final_review'].astype(str).tolist()
        
        try:
            # Generate Embeddings (Batch)
            # title parameter is optional but good practice for retrieval
            embeddings = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=texts,
                task_type="retrieval_document"
            )
            
            embedding_vectors = embeddings['embedding']

            for index, (text, vector) in enumerate(zip(texts, embedding_vectors)):
                # Map back to original row using implicit order (valid_batch is aligned with texts)
                row = valid_batch.iloc[index]
                
                # Create Document
                doc = {
                    "review_text": text,
                    "embedding": vector,
                    "model": row.get("model", "Unknown"),
                    "sentiment_label": row.get("sentiment_label", "Neutral"),
                    "sentiment_score": float(row.get("sentiment_score", 0.0)),
                    "original_index": int(i + index)
                }
                
                # Upsert based on original_index (or unique review signature if available)
                # Using original_index + model as unique constraint proxy
                filter_query = {"original_index": doc["original_index"], "model": doc["model"]}
                operations.append(UpdateOne(filter_query, {"$set": doc}, upsert=True))

        except Exception as e:
            logger.error(f"⚠️ Error batch {i}: {e}")
            time.sleep(2) # Backoff

        # Execute Bulk Write every batch
        if operations:
            collection.bulk_write(operations)
            operations = []
            
    logger.info(f"✅ Successfully ingested {total_reviews} documents into '{COLLECTION_NAME}'.")
    logger.info("ℹ️ Next Step: Create the Vector Search Index in MongoDB Atlas.")

if __name__ == "__main__":
    ingest_data()
