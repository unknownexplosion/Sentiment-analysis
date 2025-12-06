import os
import re
import json
import logging
import pandas as pd
import google.generativeai as genai
from pymongo import MongoClient
from datetime import datetime

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import certifi

# ...

class GenAIAnalyzer:
    def __init__(self, google_api_key=None, mongo_uri=None):
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY is not set. GenAI features will not work.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')

        if not self.mongo_uri:
            logger.warning("MONGO_URI is not set. Database features will not work.")
            self.db = None
        else:
            try:
                # Add tlsCAFile to fix SSL error on Mac/Streamlit Cloud
                # Added tlsAllowInvalidCertificates=True to bypass strict firewall/proxy SSL modifications
                self.client = MongoClient(self.mongo_uri, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
                self.db = self.client.get_database("sentiment_analysis_db") # Default DB name
                logger.info("Connected to MongoDB.")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                self.db = None



    def retrieve_context(self, model_name, queries, k=15):
        """
        Retrieves relevant reviews from MongoDB using Vector Search.
        """
        if self.db is None:
            return []

        # Generate embeddings for the queries
        # We aggregate multiple aspect queries into a rich context
        context_reviews = []
        seen_ids = set()

        try:
            for query in queries:
                # 1. Embed Query
                q_embedding = genai.embed_content(
                    model='models/text-embedding-004',
                    content=query,
                    task_type="retrieval_query"
                )['embedding']

                # 2. Vector Search Pipeline
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "embedding",
                            "queryVector": q_embedding,
                            "numCandidates": 100,
                            "limit": k,
                            "filter": {"model": {"$eq": model_name}}
                        }
                    },
                    {
                        "$project": {
                            "review_text": 1,
                            "sentiment_label": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
                
                cursor = self.db["review_embeddings"].aggregate(pipeline)
                
                for doc in cursor:
                    # Deduplicate based on text hash or id if available
                    # Using text hash for simplicity
                    doc_hash = hash(doc['review_text'])
                    if doc_hash not in seen_ids:
                        context_reviews.append({
                            "review_text": doc['review_text'],
                            "overall_sentiment": doc.get('sentiment_label', 'Neutral')
                        })
                        seen_ids.add(doc_hash)
            
            return context_reviews

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def generate_report(self, model_name, reviews_df_unused=None):
        """
        RAG-Enabled Report Generation.
        Note: reviews_df_unused is kept for signature compatibility but we use DB source now.
        """
        if not self.api_key:
            return {"error": "Missing Google API Key"}

        # 1. Retrieve Context (The "R" in RAG)
        # We explicitly search for key problem areas to ensure the model sees them
        queries = [
            f"Reviews about {model_name} battery life and draining",
            f"Reviews about {model_name} camera quality photos",
            f"Reviews about {model_name} performance speed lag",
            f"Reviews about {model_name} display screen brightness",
            f"General pros and cons of {model_name}"
        ]
        
        logger.info("🔍 Retrieving vector context from MongoDB...")
        rag_context = self.retrieve_context(model_name, queries)
        
        # Fallback if DB is empty or search fails (e.g., Index not built yet)
        if not rag_context and reviews_df_unused is not None:
             logger.warning("⚠️ Vector Search returned 0 results. Falling back to simple sampling.")
             rag_context = reviews_df_unused[['final_review', 'sentiment_label']].rename(columns={'final_review': 'review_text', 'sentiment_label': 'overall_sentiment'}).head(50).to_dict(orient='records')
        
        logger.info(f"✅ Retrieved {len(rag_context)} unique relevant reviews for context.")

        # Construct the Prompt with RAG Context
        # Reduced from 60 to 40 to prevent JSON response truncation (Token Limit)
        absa_json_str = json.dumps(rag_context[:40])

        prompt = f"""
        You are an expert Product Analyst and Data Engineer.
        
        You are given:
        1. The device model name.
        2. A list of RELEVANT reviews retrieved via Vector Search (Raw Text & Overall Sentiment).

        Your tasks:
        A. Read all reviews and PERFORM Aspect-Based Sentiment Analysis (ABSA) on them:
           - Identify aspects like battery, camera, display, performance, price.
           - Determine sentiment for each aspect.
        
        B. Write a manufacturer-focused report:
           - Summarize key strengths and weaknesses per aspect.
           - Highlight recurring issues or complaints found in this retrieved evidence.
           - Suggest 3–7 concrete, actionable improvements for the next hardware/software iteration.
           - Use clear, professional language.

        C. Produce output in the following EXACT JSON structure so it can be saved to MongoDB Atlas:

        {{
          "manufacturer_report": {{
            "model": "{model_name}",
            "report": "<string: detailed manufacturer-focused summary>"
          }},
          "review_aspect_records": [
            {{
              "model": "{model_name}",
              "review_id": "<string: generate unique id>",
              "review_text": "<string: original review text>",
              "overall_sentiment": "<string: one of: positive, negative, neutral>",
              "aspects": [
                {{
                  "name": "<string: aspect name>",
                  "sentiment": "<string: sentiment>",
                  "confidence": <number between 0.5 and 1.0>,
                  "evidence_span": "<snippet>"
                }}
              ]
            }}
          ]
        }}

        Rules:
        - Keep the JSON valid.
        - Escape any quotes within strings properly.
        - Check for trailing commas (do NOT include them).
        - Every review in the input must appear once in "review_aspect_records".
        - "model" must be the SAME for all records.
        - RETURN ONLY JSON.

        Now use the following input:

        MODEL_NAME:
        {model_name}

        RETRIEVED_EVIDENCE (JSON):
        {absa_json_str}
        """

        try:
            logger.info(f"Sending prompt to Gemini for {model_name}...")
            
            # Enforce JSON output mode and increase token limit to prevent truncation
            generation_config = {
                "response_mime_type": "application/json",
                "max_output_tokens": 8192,
                "temperature": 0.2
            }
            
            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config
            )
            
            text_response = self._clean_json_output(response.text)
            result_json = json.loads(text_response)
            return result_json

        except Exception as e:
            logger.error(f"GenAI generation failed: {e}")
            return {"error": str(e)}

    def _clean_json_output(self, text):
        """Standardizes JSON string to prevent parsing errors."""
        text = text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def save_to_db(self, data):
        """
        Saves the manufacturer report and aspect records to MongoDB.
        """
        if self.db is None:
            return False, "Database not connected."
        
        try:
            # 1. Save Report
            if "manufacturer_report" in data:
                reports_col = self.db["manufacturer_reports"]
                report_data = data["manufacturer_report"]
                report_data["created_at"] = datetime.utcnow()
                reports_col.insert_one(report_data)
            
            # 2. Save Records
            if "review_aspect_records" in data:
                records_col = self.db["review_aspects"]
                records = data["review_aspect_records"]
                if records:
                    records_col.insert_many(records)
            
            return True, "Successfully saved to MongoDB."
        
        except Exception as e:
            logger.error(f"MongoDB write failed: {e}")
            return False, str(e)
