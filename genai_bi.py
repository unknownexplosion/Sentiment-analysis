import os
import json
import logging
import time
import toml
import pandas as pd
import pymongo
import certifi
from pathlib import Path

# New Google Gen AI SDK (replaces deprecated google-generativeai)
from google import genai
from google.genai import types

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GenAI_BI")

# --- Configuration & Secrets ---
def load_secrets():
    """Load secrets from env or .streamlit/secrets.toml"""
    g_key = os.getenv("GOOGLE_API_KEY")
    m_uri = os.getenv("MONGO_URI")

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

# Initialise the new SDK client
_genai_client = None
if GOOGLE_API_KEY:
    _genai_client = genai.Client(api_key=GOOGLE_API_KEY)


# --- Fair Prompt Data Builder (v2 — confidence-weighted, token-efficient) ---
def build_fair_prompt_data(df_model: pd.DataFrame, sample_cap: int = 50) -> dict:
    """
    Builds a statistically representative, token-efficient summary so ALL
    records influence the report.

    v2 improvements over v1:
    ─────────────────────────────────────────────────────────────────────────
    1. Confidence-weighted sentiment — uncertain predictions contribute less
    2. Data quality metrics — avg confidence, low-conf count for LLM caveats
    3. Pre-extracted extremes — top 5 pain points + top 5 praises by confidence
    4. Truncated sample text — 150 char cap to save tokens
    5. Sample cap 80→50 — stats carry the weight, samples are just for color
    """
    total_records = len(df_model)
    df = df_model.copy()

    # Ensure confidence column exists (graceful fallback)
    if "confidence" not in df.columns:
        df["confidence"] = 0.85

    # ── 1. Data quality metrics (uses ALL records) ───────────────────────
    avg_confidence = round(df["confidence"].mean(), 3)
    low_conf_count = int((df["confidence"] < 0.6).sum())
    low_conf_pct = round(low_conf_count / total_records * 100, 1) if total_records else 0

    # ── 2. Sentiment distribution — raw AND confidence-weighted ──────────
    df["label_lower"] = df["label"].str.lower()

    # Raw distribution (simple count-based)
    raw_dist = (
        df["label_lower"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .to_dict()
    )

    # Confidence-weighted distribution
    # A 98%-confident "Positive" contributes 0.98 weight; a 52%-confident one → 0.52
    weighted = df.groupby("label_lower")["confidence"].sum()
    total_weight = weighted.sum()
    weighted_dist = (
        (weighted / total_weight * 100).round(1).to_dict()
        if total_weight > 0 else raw_dist
    )

    # ── 3. Per-aspect breakdown (compact, confidence-aware) ──────────────
    aspect_stats = {}
    for aspect, group in df.groupby("aspect"):
        n = len(group)
        pos = (group["label"] == "Positive").sum()
        neg = (group["label"] == "Negative").sum()
        neu = (group["label"] == "Neutral").sum()
        avg_conf = round(group["confidence"].mean(), 3)
        aspect_stats[aspect] = {
            "n": n,
            "pos_pct": round(pos / n * 100, 1),
            "neg_pct": round(neg / n * 100, 1),
            "neu_pct": round(neu / n * 100, 1),
            "avg_conf": avg_conf,
        }

    # Sort by volume, keep top 15
    aspect_stats = dict(
        sorted(aspect_stats.items(), key=lambda x: -x[1]["n"])[:15]
    )

    # ── 4. Pre-extracted extremes (most actionable, saves LLM hunting) ───
    # Top 5 most confidently NEGATIVE reviews → strongest pain points
    neg_df = df[df["label"] == "Negative"].nlargest(5, "confidence")
    pain_points = [
        {
            "text": row["text"][:200],
            "aspect": row["aspect"],
            "confidence": round(row["confidence"], 3),
        }
        for _, row in neg_df.iterrows()
    ]

    # Top 5 most confidently POSITIVE reviews → strongest praise
    pos_df = df[df["label"] == "Positive"].nlargest(5, "confidence")
    top_praise = [
        {
            "text": row["text"][:200],
            "aspect": row["aspect"],
            "confidence": round(row["confidence"], 3),
        }
        for _, row in pos_df.iterrows()
    ]

    # ── 5. Stratified sample (truncated text, token-efficient) ───────────
    def stratified_sample(group):
        n_samples = max(1, int(sample_cap * len(group) / total_records)) if total_records > 0 else 1
        n = min(len(group), n_samples)
        return group.sample(
            n=n,
            random_state=42,
        )

    sampled = (
        df.groupby(["aspect", "label"], group_keys=False)
        .apply(stratified_sample)
        .sample(frac=1, random_state=42)
        .head(sample_cap)
    )

    # Truncate text to 150 chars to save tokens — aspect + label already
    # tell the LLM what category it is; text just provides narrative flavor
    sampled_reviews = [
        {
            "text": row["text"][:150],
            "aspect": row["aspect"],
            "sentiment": row["label"],
            "conf": round(row["confidence"], 2),
        }
        for _, row in sampled.iterrows()
    ]

    return {
        "total_records": total_records,
        "data_quality": {
            "avg_model_confidence": avg_confidence,
            "low_confidence_predictions": low_conf_count,
            "low_confidence_pct": low_conf_pct,
        },
        "sentiment_pct": raw_dist,
        "confidence_weighted_sentiment_pct": weighted_dist,
        "aspect_breakdown": aspect_stats,
        "top_5_pain_points": pain_points,
        "top_5_strongest_praise": top_praise,
        "stratified_sample": sampled_reviews,
    }


# --- Core Class ---
class BISummarizer:
    # Model to use - gemini-2.5-flash is stable and available
    MODEL = "gemini-2.5-flash"

    def __init__(self):
        if not GOOGLE_API_KEY:
            logger.error("Google API Key missing.")
            raise ValueError("Google API Key missing.")
        if _genai_client is None:
            raise ValueError("Gemini client could not be initialised.")
        self.client = _genai_client
        self._mongo = pymongo.MongoClient(
            MONGO_URI,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=10000,
        ) if MONGO_URI else None

    def generate_for_model(self, model_name: str, df_model: pd.DataFrame):
        """
        Generates a structured BI summary JSON for a single model.
        Uses build_fair_prompt_data to ensure ALL records influence the report.
        Retries on 429 / quota errors with exponential back-off.
        """
        if df_model.empty:
            logger.warning(f"No ABSA records for {model_name}.")
            return None

        # Build statistically fair prompt data from ALL records
        fair_data = build_fair_prompt_data(df_model)
        fair_data_str = json.dumps(fair_data, indent=2, default=str)

        prompt = f"""You are a senior BI analyst at a consumer electronics company.

Product: {model_name}
Total reviews analysed: {fair_data['total_records']}
Model confidence: avg {fair_data['data_quality']['avg_model_confidence']}, {fair_data['data_quality']['low_confidence_pct']}% low-confidence predictions

DATA (all statistics computed from ALL {fair_data['total_records']} reviews):
{fair_data_str}

INSTRUCTIONS:
- Use the pre-computed statistics as ground truth — do NOT re-estimate percentages from the sample
- Use "confidence_weighted_sentiment_pct" (not raw counts) when reporting overall sentiment
- Reference "top_5_pain_points" and "top_5_strongest_praise" for specific customer quotes
- If low_confidence_pct > 15%, add a caveat in the executive overview about prediction uncertainty
- The "stratified_sample" is for qualitative color only — do not count or percentagise from it

Return this exact JSON structure:
{{
  "model": "{model_name}",
  "business_summary": {{
    "executive_overview": "2-3 sentence summary citing exact percentages from the data",
    "key_strengths": [
      {{"aspect": "...", "summary": "...", "supporting_sentiment": {{"positive_pct": ..., "review_count": ...}}}}
    ],
    "key_issues": [
      {{"aspect": "...", "priority": "HIGH|MEDIUM|LOW", "summary": "...", "supporting_sentiment": {{"negative_pct": ..., "review_count": ...}}}}
    ],
    "recommendations": [
      {{"title": "...", "description": "...", "linked_aspects": [...], "expected_impact": "..."}}
    ]
  }}
}}
Return ONLY valid JSON, no markdown fences."""

        max_retries = 4
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling Gemini for '{model_name}' (attempt {attempt + 1}/{max_retries})...")
                response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.3,
                    ),
                )
                text = response.text.strip()
                # Strip any accidental markdown fences
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text)

            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    wait = 15 * (2 ** attempt)  # 15s, 30s, 60s, 120s
                    logger.warning(f"Rate limit hit. Waiting {wait}s before retry...")
                    time.sleep(wait)
                elif "API_KEY" in err or "permission" in err.lower():
                    logger.error(f"Auth error — check GOOGLE_API_KEY: {e}")
                    return None
                else:
                    logger.error(f"Gemini call failed: {e}")
                    return None

        logger.error(f"All {max_retries} attempts failed for '{model_name}'.")
        return None

    def save_to_mongodb(self, summary_json: dict) -> bool:
        """Upserts the BI summary into MongoDB."""
        if getattr(self, '_mongo', None) is None or not summary_json:
            return False

        try:
            db = self._mongo.get_database("sentiment_analysis_db")
            col = db["manufacturer_bi_summaries"]

            model_name = summary_json.get("model")
            col.update_one(
                {"model": model_name},
                {"$set": summary_json},
                upsert=True,
            )
            logger.info(f"✅ Saved BI Summary for '{model_name}' to MongoDB.")
            return True
        except Exception as e:
            logger.error(f"MongoDB save failed: {e}")
            return False


# --- CLI Execution ---
if __name__ == "__main__":
    absa_path = "outputs/absa_training_dataset.csv"
    if not os.path.exists(absa_path):
        logger.error("ABSA Dataset not found. Run sentiment_pipeline.py first.")
        exit()

    df = pd.read_csv(absa_path)
    bi_bot = BISummarizer()

    models = df["model_name"].unique()
    logger.info(f"Found {len(models)} models to process.")

    for model in models:
        df_model = df[df["model_name"] == model]

        summary_json = bi_bot.generate_for_model(model, df_model)
        if summary_json:
            bi_bot.save_to_mongodb(summary_json)
        else:
            logger.error(f"Failed to generate summary for '{model}' after all retries.")
