import os
import json
import logging
import toml
import pandas as pd
import google.generativeai as genai
import pymongo
import certifi
from pathlib import Path

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GenAI_BI")

# --- Configuration & Secrets ---
def load_secrets():
    """Load secrets from .env or .streamlit/secrets.toml"""
    # 1. Env Vars
    g_key = os.getenv("GOOGLE_API_KEY")
    m_uri = os.getenv("MONGO_URI")
    
    # 2. Streamlit Secrets (Fallback)
    if not g_key or not m_uri:
        try:
            secrets_path = Path(".streamlit/secrets.toml")
            if secrets_path.exists():
                secrets = toml.load(secrets_path)
                g_key = g_key or secrets.get("GOOGLE_API_KEY") or secrets.get("general", {}).get("GOOGLE_API_KEY")
                m_uri = m_uri or secrets.get("MONGO_URI") or secrets.get("general", {}).get("MONGO_URI")
        except Exception as e:
            logger.warning(f"Could not load secrets.toml: {e}")
            
    return g_key, m_uri

GOOGLE_API_KEY, MONGO_URI = load_secrets()

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Core Class ---
class BISummarizer:
    def __init__(self):
        if not GOOGLE_API_KEY:
            logger.error("Google API Key missing.")
            return
        
        # Available model confirmed via debug: gemini-2.0-flash
        self.model = genai.GenerativeModel('gemini-2.0-flash') 

    def generate_for_model(self, model_name, absa_records):
        """
        Generates the BI summary for a single model using the provided template.
        Includes internal retry logic for 429 errors.
        """
        import time
        
        # 1. Validate Input
        if not absa_records:
            logger.warning(f"No ABSA records for {model_name}.")
            return None

        # 2. construct JSON for prompt
        prompt_json = []
        for i, r in enumerate(absa_records):
            prompt_json.append({
                "review_id": i, 
                "review_text": r.get('text', ''),
                "aspect": r.get('aspect', 'General'),
                "sentiment": r.get('label', 'Neutral').lower(),
                "confidence": 0.95 
            })
            
        absa_json_str = json.dumps(prompt_json, indent=2)

        # 3. Build the Prompt
        prompt = f"""
        You are a senior Business Intelligence analyst in a consumer electronics company.
        
        Model: {model_name}
        
        ABSA JSON:
        {absa_json_str}
        
        Task: Analyze ABSA data and return a JSON summary (Executive Overview, Strengths, Issues, Recommendations).
        
        Output JSON Structure:
        {{
          "model": "{model_name}",
          "business_summary": {{
            "executive_overview": "...",
            "key_strengths": [{{ "aspect": "...", "summary": "...", "supporting_sentiment": {{ ... }} }}],
            "key_issues": [{{ "aspect": "...", "priority": "HIGH", "summary": "...", "supporting_sentiment": {{ ... }} }}],
            "recommendations": [{{ "title": "...", "description": "...", "linked_aspects": [], "expected_impact": "..." }}]
          }}
        }}
        """

        # 4. Call Model with Retry
        max_retries = 3
        base_wait = 10
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending prompt to Gemini for {model_name} (Attempt {attempt+1})...")
                response = self.model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text)
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "Quota" in error_str:
                    wait_time = base_wait * (2 ** attempt) # Exponential backoff
                    logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gemini call failed: {e}")
                    return None
        
        return None

    def save_to_mongodb(self, summary_json):
        """Saves the summary to DB1."""
        if not MONGO_URI or not summary_json:
            return False
        
        try:
            client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
            db = client.get_database("sentiment_analysis_db")
            col = db["manufacturer_bi_summaries"] # DB1
            
            # Upsert based on model name
            model_name = summary_json.get("model")
            col.update_one(
                {"model": model_name},
                {"$set": summary_json},
                upsert=True
            )
            logger.info(f"✅ Saved BI Summary for {model_name} to MongoDB.")
            return True
        except Exception as e:
            logger.error(f"MongoDB save failed: {e}")
            return False

# --- CLI Execution ---
if __name__ == "__main__":
    # 1. Load Data
    absa_path = "outputs/absa_training_dataset.csv"
    if not os.path.exists(absa_path):
        logger.error("ABSA Dataset not found. Run sentiment_pipeline.py first.")
        exit()
        
    df = pd.read_csv(absa_path)
    
    bi_bot = BISummarizer()
    
    import time

    # 2. Iterate Models
    models = df['model_name'].unique()
    logger.info(f"Found {len(models)} models to process.")
    
    for model in models:
        model_records = df[df['model_name'] == model].to_dict('records')
        
        # Retry Logic
        max_retries = 3
        for attempt in range(max_retries):
            summary_json = bi_bot.generate_for_model(model, model_records[:100])
            
            if summary_json:
                bi_bot.save_to_mongodb(summary_json)
                logger.info("Waiting 60s to respect rate limits...")
                time.sleep(60) # Prevent 429
                break
            else:
                logger.warning(f"Attempt {attempt+1} failed for {model}. Retrying in 60s...")
                time.sleep(60)
