"""
reddit_scraper.py
==================
Automated Reddit scraper for Apple product sentiment data.

Uses Reddit's public JSON API — NO credentials or API key required.
Hits endpoints like: https://www.reddit.com/r/apple/hot.json
with a browser-like User-Agent header. Fully compliant with Reddit's
public data access policy for non-commercial research.

Run manually:
  python reddit_scraper.py

Or schedule weekly (see weekly_scheduler.py).
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RedditScraper")

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("outputs/scraped")
STATE_FILE  = OUTPUT_DIR / "scraper_state.json"
MERGED_CSV  = OUTPUT_DIR / "reddit_reviews_all.csv"

# Subreddits to scrape — curated Apple communities
TARGET_SUBREDDITS = [
    # General
    "apple",
    "iPhone",
    "Mac",
    "iPad",
    "MacOS",
    "AppleWatch",
    "AirPods",
    "AppleMusic",
    # Buying & help
    "appleswap",
    "applehelp",
    "AppleWhatShouldIBuy",
    "iphonehelp",
    "AppleIndia",
    # Developer & opinion
    "iOSProgramming",
    "applesucks",
]

# Feeds to pull from each subreddit
FEEDS = ["hot", "new", "top"]

# Posts limit per feed per subreddit (Reddit public JSON max = 100)
POSTS_PER_FEED = 40

# Keyword → canonical product name mapping
# Keyword → canonical product name mapping
# Ordered from most specific to most generic so the first match wins
MODEL_KEYWORDS = {
    # ── iPhone ──────────────────────────────────────────────────────────────
    "iPhone 17e":           ["iphone 17e"],
    "iPhone 16 Pro Max":    ["iphone 16 pro max"],
    "iPhone 16 Pro":        ["iphone 16 pro"],
    "iPhone 16 Plus":       ["iphone 16 plus"],
    "iPhone 16e":           ["iphone 16e"],
    "iPhone 16":            ["iphone 16"],
    "iPhone 15 Pro Max":    ["iphone 15 pro max"],
    "iPhone 15 Pro":        ["iphone 15 pro"],
    "iPhone 15 Plus":       ["iphone 15 plus"],
    "iPhone 15":            ["iphone 15"],
    "iPhone 14 Pro Max":    ["iphone 14 pro max"],
    "iPhone 14 Pro":        ["iphone 14 pro"],
    "iPhone 14 Plus":       ["iphone 14 plus"],
    "iPhone 14":            ["iphone 14"],
    "iPhone 13 Pro Max":    ["iphone 13 pro max"],
    "iPhone 13 Pro":        ["iphone 13 pro"],
    "iPhone 13 mini":       ["iphone 13 mini"],
    "iPhone 13":            ["iphone 13"],
    "iPhone 12 Pro Max":    ["iphone 12 pro max"],
    "iPhone 12 Pro":        ["iphone 12 pro"],
    "iPhone 12 mini":       ["iphone 12 mini"],
    "iPhone 12":            ["iphone 12"],
    "iPhone 11 Pro Max":    ["iphone 11 pro max"],
    "iPhone 11 Pro":        ["iphone 11 pro"],
    "iPhone 11":            ["iphone 11"],
    "iPhone XS Max":        ["iphone xs max"],
    "iPhone XS":            ["iphone xs"],
    "iPhone XR":            ["iphone xr"],
    "iPhone X":             ["iphone x"],
    "iPhone 8 Plus":        ["iphone 8 plus"],
    "iPhone 8":             ["iphone 8"],
    "iPhone 7 Plus":        ["iphone 7 plus"],
    "iPhone 7":             ["iphone 7"],
    "iPhone 6s Plus":       ["iphone 6s plus"],
    "iPhone 6s":            ["iphone 6s"],
    "iPhone 6 Plus":        ["iphone 6 plus"],
    "iPhone 6":             ["iphone 6"],
    "iPhone 5s":            ["iphone 5s"],
    "iPhone 5c":            ["iphone 5c"],
    "iPhone 5":             ["iphone 5"],
    "iPhone 4S":            ["iphone 4s"],
    "iPhone 4":             ["iphone 4"],
    "iPhone 3GS":           ["iphone 3gs"],
    "iPhone 3G":            ["iphone 3g"],
    "iPhone SE (3rd generation)": ["iphone se 3", "iphone se 3rd gen", "iphone se 2022"],
    "iPhone SE (2nd generation)": ["iphone se 2", "iphone se 2nd gen", "iphone se 2020"],
    "iPhone SE (1st generation)": ["iphone se 1", "iphone se 1st gen", "iphone se 2016"],
    "iPhone SE":            ["iphone se"],
    "iPhone (1st generation)": ["iphone 2g", "original iphone", "iphone 1st gen"],

    # ── MacBook Air ─────────────────────────────────────────────────────────
    "MacBook Air (M5, 15-inch)": ["macbook air m5 15", "m5 15-inch air", "m5 15 inch air"],
    "MacBook Air (M5, 13-inch)": ["macbook air m5 13", "m5 13-inch air", "m5 13 inch air", "macbook air m5"],
    "MacBook Air (M4)":          ["macbook air m4", "m4 air"],
    "MacBook Air (M3, 15-inch)": ["macbook air m3 15", "m3 15-inch air", "m3 15 inch air"],
    "MacBook Air (M3, 13-inch)": ["macbook air m3 13", "m3 13-inch air", "m3 13 inch air", "macbook air m3"],
    "MacBook Air (M2, 15-inch)": ["macbook air m2 15", "m2 15-inch air", "m2 15 inch air"],
    "MacBook Air (M2, 13-inch)": ["macbook air m2 13", "m2 13-inch air", "m2 13 inch air", "macbook air m2"],
    "MacBook Air (M1)":          ["macbook air m1", "m1 air"],
    "MacBook Air (Retina)":      ["macbook air retina", "macbook air 2018", "macbook air 2019"],
    "MacBook Air (13-inch)":     ["macbook air 13", "13-inch macbook air", "13 inch macbook air"],
    "MacBook Air (11-inch)":     ["macbook air 11", "11-inch macbook air", "11 inch macbook air"],
    "MacBook Air (Original)":    ["macbook air 2008", "macbook air 2009", "original macbook air"],
    "MacBook Air":               ["macbook air", "mba"],

    # ── MacBook Pro ─────────────────────────────────────────────────────────
    "MacBook Pro (M5 Max, 16-inch)": ["macbook pro m5 max 16", "m5 max 16-inch pro", "m5 max 16 inch pro"],
    "MacBook Pro (M5 Max, 14-inch)": ["macbook pro m5 max 14", "m5 max 14-inch pro", "m5 max 14 inch pro"],
    "MacBook Pro (M5 Pro, 16-inch)": ["macbook pro m5 pro 16", "m5 pro 16-inch pro", "m5 pro 16 inch pro"],
    "MacBook Pro (M5 Pro, 14-inch)": ["macbook pro m5 pro 14", "m5 pro 14-inch pro", "m5 pro 14 inch pro"],
    "MacBook Pro (M4 Max, 16-inch)": ["macbook pro m4 max 16", "m4 max 16-inch pro"],
    "MacBook Pro (M4 Max, 14-inch)": ["macbook pro m4 max 14", "m4 max 14-inch pro"],
    "MacBook Pro (M4 Pro, 16-inch)": ["macbook pro m4 pro 16", "m4 pro 16-inch pro"],
    "MacBook Pro (M4 Pro, 14-inch)": ["macbook pro m4 pro 14", "m4 pro 14-inch pro"],
    "MacBook Pro (M4, 14-inch)":     ["macbook pro m4 14", "m4 14-inch pro", "macbook pro m4"],
    "MacBook Pro (M3 Max, 16-inch)": ["macbook pro m3 max 16", "m3 max 16-inch pro"],
    "MacBook Pro (M3 Max, 14-inch)": ["macbook pro m3 max 14", "m3 max 14-inch pro"],
    "MacBook Pro (M3 Pro, 16-inch)": ["macbook pro m3 pro 16", "m3 pro 16-inch pro"],
    "MacBook Pro (M3 Pro, 14-inch)": ["macbook pro m3 pro 14", "m3 pro 14-inch pro"],
    "MacBook Pro (M3, 14-inch)":     ["macbook pro m3 14", "m3 14-inch pro", "macbook pro m3"],
    "MacBook Pro (M2 Max, 16-inch)": ["macbook pro m2 max 16", "m2 max 16-inch pro"],
    "MacBook Pro (M2 Max, 14-inch)": ["macbook pro m2 max 14", "m2 max 14-inch pro"],
    "MacBook Pro (M2 Pro, 16-inch)": ["macbook pro m2 pro 16", "m2 pro 16-inch pro"],
    "MacBook Pro (M2 Pro, 14-inch)": ["macbook pro m2 pro 14", "m2 pro 14-inch pro"],
    "MacBook Pro (M2, 13-inch)":     ["macbook pro m2 13", "m2 13-inch pro", "macbook pro m2"],
    "MacBook Pro (M1 Max, 16-inch)": ["macbook pro m1 max 16", "m1 max 16-inch pro"],
    "MacBook Pro (M1 Max, 14-inch)": ["macbook pro m1 max 14", "m1 max 14-inch pro"],
    "MacBook Pro (M1 Pro, 16-inch)": ["macbook pro m1 pro 16", "m1 pro 16-inch pro"],
    "MacBook Pro (M1 Pro, 14-inch)": ["macbook pro m1 pro 14", "m1 pro 14-inch pro"],
    "MacBook Pro (M1, 13-inch)":     ["macbook pro m1 13", "m1 13-inch pro", "macbook pro m1"],
    "MacBook Pro (16-inch, Intel)":  ["macbook pro 16 intel", "macbook pro 16-inch 2019", "macbook pro 16 inch 2019"],
    "MacBook Pro (Touch Bar)":       ["macbook pro touch bar", "macbook pro 2016", "macbook pro 2017", "macbook pro 2018", "macbook pro 2019"],
    "MacBook Pro (Retina)":          ["macbook pro retina", "macbook pro 2012", "macbook pro 2013", "macbook pro 2014", "macbook pro 2015"],
    "MacBook Pro (Unibody)":         ["macbook pro unibody", "macbook pro 2008", "macbook pro 2009", "macbook pro 2010", "macbook pro 2011"],
    "MacBook Pro (Original)":        ["macbook pro 2006", "macbook pro 2007", "original macbook pro"],
    "MacBook Pro":                   ["macbook pro", "mbp"],

    # ── Other Mac Laptops ───────────────────────────────────────────────────
    "MacBook Neo":                   ["macbook neo"],
    "MacBook (Retina, 12-inch)":     ["macbook retina 12", "macbook 12-inch", "12 inch macbook"],
    "MacBook (Polycarbonate)":       ["macbook polycarbonate", "white macbook", "black macbook"],
    "MacBook":                       ["macbook"], # Needs to be below air/pro so it doesn't match first
    "iBook":                         ["ibook", "ibook g3", "ibook g4"],
    "PowerBook":                     ["powerbook", "powerbook g3", "powerbook g4"],
    "Macintosh Portable":            ["macintosh portable"],

    # ── Mac Desktops ────────────────────────────────────────────────────────
    "Mac Pro":              ["mac pro"],
    "Mac Studio":           ["mac studio"],
    "Mac mini":             ["mac mini"],
    "iMac":                 ["imac"],

    # ── iPad ────────────────────────────────────────────────────────────────
    "iPad Pro":             ["ipad pro"],
    "iPad Air":             ["ipad air"],
    "iPad mini":            ["ipad mini"],
    "iPad":                 ["ipad"],

    # ── Apple Watch ─────────────────────────────────────────────────────────
    "Apple Watch Ultra":    ["watch ultra", "apple watch ultra"],
    "Apple Watch SE":       ["watch se", "apple watch se"],
    "Apple Watch":          ["apple watch", "watch series"],

    # ── AirPods ─────────────────────────────────────────────────────────────
    "AirPods Max 2":        ["airpods max 2"],
    "AirPods Max":          ["airpods max"],
    "AirPods Pro 3":        ["airpods pro 3"],
    "AirPods Pro 2":        ["airpods pro 2", "airpods pro second", "airpods pro 2nd"],
    "AirPods Pro 1":        ["airpods pro 1", "airpods pro first", "airpods pro 1st"],
    "AirPods Pro":          ["airpods pro"],
    "AirPods 4 with Active Noise Cancellation": ["airpods 4 anc", "airpods 4 with active noise cancellation"],
    "AirPods 4":            ["airpods 4", "airpods fourth", "airpods 4th"],
    "AirPods 3":            ["airpods 3", "airpods third", "airpods 3rd"],
    "AirPods 2":            ["airpods 2", "airpods second", "airpods 2nd"],
    "AirPods 1":            ["airpods 1", "airpods first", "airpods 1st"],
    "AirPods":              ["airpods"],

    # ── Home & TV ───────────────────────────────────────────────────────────
    "HomePod mini":         ["homepod mini"],
    "HomePod":              ["homepod"],
    "Apple TV 4K":          ["apple tv 4k"],
    "Apple TV HD":          ["apple tv hd"],
    "Apple TV":             ["apple tv"],

    # ── Accessories ─────────────────────────────────────────────────────────
    "Vision Pro":           ["vision pro", "apple vision"],
    "AirTag":               ["airtag"],
    "Studio Display":       ["studio display"],
    "Pro Display XDR":      ["pro display xdr"],
    "Apple Pencil":         ["apple pencil"],
    "Magic Keyboard":       ["magic keyboard"],
    "Magic Mouse":          ["magic mouse", "magic mouse 2"],
    "Magic Trackpad":       ["magic trackpad"],

    # ── Services & Software ─────────────────────────────────────────────────
    "Apple Music":          ["apple music"],
    "Apple TV+":            ["apple tv+", "apple tv plus"],
    "Apple Arcade":         ["apple arcade"],
    "Apple Fitness+":       ["apple fitness+", "apple fitness plus"],
    "Apple News+":          ["apple news+", "apple news plus"],
    "iCloud":               ["icloud"],
    "App Store":            ["app store"],
    "Apple One":            ["apple one"],
    "FaceTime":             ["facetime"],
    "iMessage":             ["imessage"],
    "iOS":                  ["ios 18", "ios 17", "ios 16", "ios update", "ios beta"],
    "macOS":                ["macos", "mac os", "sequoia", "sonoma", "ventura"],

    # ── Catch-all ───────────────────────────────────────────────────────────
    "Apple":                ["apple", "siri", "apple intelligence"],
}

# Browser-like User-Agent to avoid 429/403 from Reddit's CDN
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Polite delay between requests (seconds) — keeps us under Reddit's rate limit
_REQUEST_DELAY = 1.2


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_model(text: str) -> str:
    """Map post text to a product model for pipeline compatibility."""
    t = text.lower()
    for model, keywords in MODEL_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return model
    return "Apple"


def _fetch_json(url: str, params: dict = None, retries: int = 3) -> dict | None:
    """GET a Reddit JSON endpoint with retry + back-off."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                logger.warning(f"Rate-limited — waiting {wait}s…")
                time.sleep(wait)
            elif resp.status_code in (403, 404):
                logger.warning(f"HTTP {resp.status_code} for {url} — skipping")
                return None
            else:
                logger.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt+1})")
                time.sleep(3 * (attempt + 1))
        except requests.RequestException as e:
            logger.warning(f"Request error: {e} (attempt {attempt+1})")
            time.sleep(3 * (attempt + 1))
    return None


def _parse_post(post_data: dict, subreddit: str) -> dict | None:
    """Convert a raw Reddit post dict into a pipeline-compatible row."""
    title   = post_data.get("title", "").strip()
    selftext = post_data.get("selftext", "").strip()
    text    = f"{title} {selftext}".strip()

    if len(text) < 20:
        return None
    if selftext in ("[deleted]", "[removed]"):
        text = title

    created = post_data.get("created_utc", 0)
    return {
        "model":           _detect_model(text),
        "original_review": text,
        "source":          "reddit",
        "subreddit":       subreddit,
        "post_id":         post_data.get("id", ""),
        "score":           post_data.get("score", 0),
        "upvote_ratio":    post_data.get("upvote_ratio", 0),
        "num_comments":    post_data.get("num_comments", 0),
        "scraped_at":      datetime.now(timezone.utc).isoformat(),
        "created_utc":     datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else "",
        "url":             f"https://reddit.com{post_data.get('permalink', '')}",
    }


# ── Main scraper ───────────────────────────────────────────────────────────────

class RedditScraper:
    """
    No-credential Reddit scraper using public JSON feeds.
    Compatible with the existing pipeline interface.
    """

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
        return {"seen_ids": [], "last_run": None, "run_count": 0, "total_scraped": 0}

    def _save_state(self, state: dict):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def scrape_subreddit(self, subreddit: str, seen_ids: set) -> list[dict]:
        """Scrape hot/new/top feeds for one subreddit via public JSON."""
        rows = []
        base = f"https://www.reddit.com/r/{subreddit}"

        for feed in FEEDS:
            url     = f"{base}/{feed}.json"
            params  = {"limit": POSTS_PER_FEED, "raw_json": 1}
            if feed == "top":
                params["t"] = "week"   # top posts of the week

            data = _fetch_json(url, params=params)
            if not data:
                continue

            for child in data.get("data", {}).get("children", []):
                post_data = child.get("data", {})
                pid = post_data.get("id", "")
                if pid in seen_ids:
                    continue
                row = _parse_post(post_data, subreddit)
                if row:
                    rows.append(row)
                    seen_ids.add(pid)

            time.sleep(_REQUEST_DELAY)

        logger.info(f"  r/{subreddit}: {len(rows)} new posts")
        return rows


    def run(self) -> pd.DataFrame | None:
        """
        Main scraping run. Returns a DataFrame in the format expected by
        sentiment_pipeline.py, or None if nothing new was found.
        """
        logger.info("=" * 60)
        logger.info(f"Reddit Scraper (no-auth) — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 60)

        state    = self._load_state()
        seen_ids = set(state.get("seen_ids", []))
        all_rows = []

        # Per-subreddit only — no global search to avoid off-topic posts
        for sr in TARGET_SUBREDDITS:
            logger.info(f"Scraping r/{sr}…")
            rows = self.scrape_subreddit(sr, seen_ids)
            all_rows.extend(rows)

        if not all_rows:
            logger.warning("No new posts found this run.")
            state["last_run"] = datetime.now(timezone.utc).isoformat()
            self._save_state(state)
            return None

        # Build DataFrame — only keep posts from our Apple subreddits (case-insensitive)
        target_lower = {s.lower() for s in TARGET_SUBREDDITS}
        new_df = pd.DataFrame(all_rows)
        new_df = new_df[new_df["subreddit"].str.lower().isin(target_lower)]

        # Merge with existing (dedup by post_id), also re-filter existing data
        if MERGED_CSV.exists():
            existing = pd.read_csv(MERGED_CSV)
            existing = existing[existing["subreddit"].str.lower().isin(target_lower)]
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["post_id"], keep="last", inplace=True)
        else:
            combined = new_df

        combined.to_csv(MERGED_CSV, index=False)

        # Weekly timestamped file
        week_file = OUTPUT_DIR / f"reddit_{datetime.now().strftime('%Y-W%V')}.csv"
        new_df.to_csv(week_file, index=False)

        # 4. Update state
        state["seen_ids"]      = list(seen_ids)[-10_000:]
        state["last_run"]      = datetime.now(timezone.utc).isoformat()
        state["total_scraped"] = len(combined)
        state["run_count"]     = state.get("run_count", 0) + 1
        self._save_state(state)

        logger.info(f"✅  Scraped {len(new_df)} new posts")
        logger.info(f"   Total in dataset: {len(combined)}")
        logger.info(f"   Saved: {MERGED_CSV}")
        logger.info(f"   Week file: {week_file}")

        return new_df


# ── Pipeline integration ───────────────────────────────────────────────────────

def run_pipeline_on_new_data(new_df: pd.DataFrame) -> bool:
    """
    After scraping, run the sentiment pipeline on new data and append
    results to the main sentiment_output.csv and ABSA dataset.
    """
    logger.info("Running sentiment pipeline on scraped data…")
    try:
        from sentiment_pipeline import (
            preprocess_reviews,
            translate_and_clean,
            handle_duplicates,
            analyze_sentiment_absa,
        )
        import sys

        df = new_df[["model", "original_review"]].copy()
        old_stdout = sys.stdout
        devnull = open(os.devnull, "w")
        try:
            sys.stdout = devnull
            df  = preprocess_reviews(df)
            df  = translate_and_clean(df)
            df  = handle_duplicates(df)
            absa_df, clf = analyze_sentiment_absa(df)
        finally:
            sys.stdout = old_stdout
            devnull.close()

        # Append to sentiment_output.csv (now ABSA format)
        out_csv = Path("outputs/sentiment_output.csv")
        if out_csv.exists() and not absa_df.empty:
            existing = pd.read_csv(out_csv)
            pd.concat([existing, absa_df], ignore_index=True).drop_duplicates(
                subset=["model", "sentence", "aspect"], keep="last"
            ).to_csv(out_csv, index=False)
        elif not absa_df.empty:
            absa_df.to_csv(out_csv, index=False)

        # Append to ABSA training dataset (aspect != "General")
        training_df = absa_df[absa_df['aspect'] != 'General'].copy() if not absa_df.empty else pd.DataFrame()
        if not training_df.empty:
            train_export = training_df.rename(columns={
                'sentence': 'text',
                'sentiment_label': 'label',
                'model': 'model_name',
            })[['text', 'aspect', 'label', 'confidence', 'model_name']]

            absa_csv = Path("outputs/absa_training_dataset.csv")
            if absa_csv.exists():
                pd.concat([pd.read_csv(absa_csv), train_export], ignore_index=True).to_csv(absa_csv, index=False)
            else:
                train_export.to_csv(absa_csv, index=False)

        n_reviews = absa_df['review_id'].nunique() if not absa_df.empty else 0
        logger.info(f"✅  Pipeline complete — {n_reviews} reviews, {len(absa_df)} ABSA rows")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False


def run_retraining_pipeline_on_new_data(new_df: pd.DataFrame) -> bool:
    """
    After scraping, run the retraining sentiment labeling pipeline (using nlptown teacher model)
    on new data and append results to outputs/absa_training_dataset.csv.
    Does NOT affect the production outputs/sentiment_output.csv.
    """
    logger.info("Running retraining-specific sentiment pipeline on scraped data (nlptown teacher)...")
    try:
        from nlpbert_labeler import label_with_nlptown
        import sys

        # Format input dataframe
        df = new_df[["model", "original_review"]].copy()

        # Suppress outputs to devnull
        old_stdout = sys.stdout
        devnull = open(os.devnull, "w")
        try:
            sys.stdout = devnull
            labeled_df = label_with_nlptown(df)
        finally:
            sys.stdout = old_stdout
            devnull.close()

        if labeled_df.empty:
            logger.info("No training data generated (likely no valid aspect clauses detected).")
            return True

        # Append to outputs/absa_training_dataset.csv
        absa_csv = Path("outputs/absa_training_dataset.csv")
        if absa_csv.exists():
            existing = pd.read_csv(absa_csv)
            # Combine and deduplicate based on text + aspect
            combined_df = pd.concat([existing, labeled_df], ignore_index=True)
            combined_df.drop_duplicates(subset=["text", "aspect"], keep="last").to_csv(absa_csv, index=False)
            logger.info(f"✅ Appended to existing {absa_csv}. Unique rows in dataset now: {len(combined_df.drop_duplicates(subset=['text', 'aspect']))}")
        else:
            absa_csv.parent.mkdir(parents=True, exist_ok=True)
            labeled_df.to_csv(absa_csv, index=False)
            logger.info(f"✅ Created new {absa_csv} with {len(labeled_df)} rows.")

        logger.info(f"✅ Retraining pipeline complete — {len(labeled_df)} ABSA training rows added.")
        return True

    except Exception as e:
        logger.error(f"Retraining pipeline failed: {e}")
        return False



# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reddit Scraper — no credentials required")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="Skip sentiment pipeline after scraping")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch 5 posts from r/apple to verify connectivity")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN — testing connectivity to r/apple…")
        data = _fetch_json("https://www.reddit.com/r/apple/hot.json", params={"limit": 5, "raw_json": 1})
        if data:
            titles = [c["data"]["title"] for c in data["data"]["children"]]
            logger.info(f"✅  Connected. Sample posts:")
            for t in titles:
                logger.info(f"   • {t[:80]}")
        else:
            logger.error("❌  Could not reach Reddit. Check your internet connection.")
    else:
        scraper = RedditScraper()
        new_df  = scraper.run()
        if new_df is not None and not args.no_pipeline:
            run_pipeline_on_new_data(new_df)
