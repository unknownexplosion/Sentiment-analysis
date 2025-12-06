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

        # Assign temporary IDs to track context without echoing full text
        # Restored to 40 reviews since we are no longer echoing text
        rag_context_subset = rag_context[:40]
        context_map = {}
        for idx, item in enumerate(rag_context_subset):
            item['id'] = idx
            context_map[idx] = item
            
        absa_json_str = json.dumps(rag_context_subset)

        prompt = f"""
        You are an expert Product Analyst and Data Engineer.
        
        You are given:
        1. The device model name.
        2. A list of RELEVANT reviews (id, review_text, sentiment).

        Your tasks:
        A. Read all reviews and PERFORM Aspect-Based Sentiment Analysis (ABSA).
        B. Write a manufacturer-focused report.
        C. Extract aspect data for each review.
        
        IMPORTANT OPTIMIZATION RULES:
        - In the "review_aspect_records" output, DO NOT return the "review_text" or "model".
        - ONLY return the "id" and the "aspects" list.
        - I will map the text back myself using the "id".

        Output JSON Structure:

        {{
          "manufacturer_report": {{
            "model": "{model_name}",
            "report": "<string: detailed manufacturer-focused summary>"
          }},
          "review_aspect_records": [
            {{
              "id": <int: matching the input id>,
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
        - "review_aspect_records" must contain one entry for every input review.
        - RETURN ONLY JSON.

        Input Data:
        MODEL_NAME: {model_name}
        EVIDENCE: {absa_json_str}
        """

        try:
            logger.info(f"Sending prompt to Gemini for {model_name}...")
            
            # Enforce JSON output mode
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
            
            # Post-processing: Re-attach original text and metadata
            if "review_aspect_records" in result_json:
                enriched_records = []
                for record in result_json["review_aspect_records"]:
                    r_id = record.get("id")
                    if r_id is not None and r_id in context_map:
                        original = context_map[r_id]
                        # Merge the LLM's aspects with the Original Content
                        full_record = {
                            "model": model_name,
                            "review_text": original.get("review_text", ""),
                            "overall_sentiment": original.get("overall_sentiment", "Neutral"),
                            "aspects": record.get("aspects", [])
                        }
                        enriched_records.append(full_record)
                
                # Replace the lightweight list with the full records
                result_json["review_aspect_records"] = enriched_records
                
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
