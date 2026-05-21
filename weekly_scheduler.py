"""
weekly_scheduler.py
====================
Weekly job scheduler that:
  1. Runs the Reddit scraper every Sunday at midnight (or on-demand)
  2. Runs the sentiment pipeline on new data
  3. Saves results to the database

Run persistently (keeps running in background):
  python weekly_scheduler.py

Or trigger once immediately:
  python weekly_scheduler.py --now

Or just check the next scheduled run:
  python weekly_scheduler.py --status
"""

import patch_transformers
import argparse
import logging
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WeeklyScheduler")

SCHEDULE_FILE = Path("outputs/scraped/schedule.json")
RUN_DAY       = 6       # Sunday (0=Monday, 6=Sunday)
RUN_HOUR      = 2       # 2:00 AM local time


def load_schedule() -> dict:
    if SCHEDULE_FILE.exists():
        with open(SCHEDULE_FILE) as f:
            return json.load(f)
    return {"last_run": None, "next_run": None, "run_count": 0}


def save_schedule(state: dict):
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEDULE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def next_sunday(hour: int = RUN_HOUR) -> datetime:
    """Calculate next occurrence of Sunday at `hour`."""
    now = datetime.now()
    days_until_sunday = (6 - now.weekday()) % 7
    if days_until_sunday == 0 and now.hour >= hour:
        days_until_sunday = 7  # Already past today's window
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    target += timedelta(days=days_until_sunday)
    return target

def next_month(day_of_month: int = 1, hour: int = RUN_HOUR) -> datetime:
    """Calculate the next occurrence of the specified day of the month."""
    now = datetime.now()
    target = now.replace(day=day_of_month, hour=hour, minute=0, second=0, microsecond=0)
    if now >= target:
        # Move to next month
        if target.month == 12:
            target = target.replace(year=target.year + 1, month=1)
        else:
            target = target.replace(month=target.month + 1)
    return target


def run_weekly_job():
    """Execute the full weekly pipeline: scrape → analyse → save."""
    logger.info("=" * 60)
    logger.info(f"⏰ Weekly Job — {datetime.now().strftime('%A %Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    state = load_schedule()

    # Step 1 — Scrape Reddit
    logger.info("Step 1/2 — Running Reddit scraper…")
    try:
        from reddit_scraper import RedditScraper, run_pipeline_on_new_data
        scraper = RedditScraper()
        new_df = scraper.run()

        # Step 2 — Run pipeline
        if new_df is not None and len(new_df) > 0:
            from reddit_scraper import run_retraining_pipeline_on_new_data
            logger.info(f"Step 2a/2 — Running production sentiment pipeline on {len(new_df)} new posts…")
            success_prod = run_pipeline_on_new_data(new_df)
            
            logger.info("Step 2b/2 — Running retraining sentiment pipeline (nlptown teacher)…")
            success_train = run_retraining_pipeline_on_new_data(new_df)
            
            if success_prod and success_train:
                logger.info("✅ Weekly job completed successfully (both pipelines run)!")
            elif success_prod:
                logger.warning("⚠️ Production pipeline succeeded but retraining pipeline failed.")
            else:
                logger.error("⚠️ Pipelines failed — scraped data saved but not fully analysed.")
        else:
            logger.info("No new data this week. Skipping pipeline.")

    except ImportError as e:
        logger.error(f"Import error: {e}. Run: pip install praw")
        return
    except ValueError as e:
        logger.error(f"Config error: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return

    # Update schedule state
    state["last_run"]  = datetime.now(timezone.utc).isoformat()
    state["next_run"]  = next_sunday().isoformat()
    state["run_count"] = state.get("run_count", 0) + 1
    save_schedule(state)
    logger.info(f"Next scheduled run: {state['next_run']}")

def _load_metrics(path: str) -> dict:
    """Safely load a metrics JSON file. Returns empty dict if not found."""
    import json
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _is_better(new: dict, old: dict) -> bool:
    """
    Return True if the new model is good enough to deploy.
    Rules (in priority order):
      1. F1 MUST improve or hold (primary metric — most robust to imbalance)
      2. At least 2 of 3 metrics (Accuracy, F1, Precision) must improve or hold
    If no baseline exists (old is empty), always returns True.
    """
    if not old:
        logger.info("📊 No baseline metrics found — treating new model as an automatic improvement.")
        return True

    checks = [
        ("eval_accuracy",  "Accuracy"),
        ("eval_f1",        "F1-Score"),
        ("eval_precision", "Precision"),
    ]
    improved_count = 0
    for key, label in checks:
        old_val = old.get(key, 0.0)
        new_val = new.get(key, 0.0)
        symbol = "✅" if new_val >= old_val else "❌"
        logger.info(f"  {symbol} {label}: {old_val:.4f} → {new_val:.4f}")
        if new_val >= old_val:
            improved_count += 1

    # Hard gate: F1 must not regress
    if new.get("eval_f1", 0) < old.get("eval_f1", 0):
        logger.warning("🚫 F1 regressed — primary gate failed. Blocking deployment.")
        return False

    # Soft gate: at least 2/3 metrics must improve or hold
    if improved_count < 2:
        logger.warning(f"🚫 Only {improved_count}/3 metrics improved — blocking deployment.")
        return False

    return True


def run_monthly_retraining_job():
    """Execute the monthly retraining pipeline: train → compare → upload if improved."""
    logger.info("=" * 60)
    logger.info(f"🧠 Monthly Retraining Job — {datetime.now().strftime('%A %Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    state = load_schedule()
    metrics_path = "outputs/fine_tuned_absa_model/metrics.json"

    # Snapshot baseline metrics BEFORE training overwrites the file
    baseline_metrics = _load_metrics(metrics_path)
    if baseline_metrics:
        logger.info(
            f"📌 Baseline — Accuracy: {baseline_metrics.get('eval_accuracy', 0):.4f}  "
            f"F1: {baseline_metrics.get('eval_f1', 0):.4f}  "
            f"Precision: {baseline_metrics.get('eval_precision', 0):.4f}"
        )
    else:
        logger.info("📌 No baseline metrics file found — first run, will upload unconditionally if training succeeds.")

    # Step 1 — Train the model
    logger.info("Step 1/3 — Running Model Training…")
    try:
        from train_absa_model import train
        train()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return

    # Step 2 — Compare new vs baseline
    logger.info("Step 2/3 — Comparing new model vs baseline…")
    new_metrics = _load_metrics(metrics_path)
    if not new_metrics:
        logger.error("⚠️ Could not read new metrics.json after training. Aborting upload.")
        return

    logger.info(
        f"🆕 New Model — Accuracy: {new_metrics.get('eval_accuracy', 0):.4f}  "
        f"F1: {new_metrics.get('eval_f1', 0):.4f}  "
        f"Precision: {new_metrics.get('eval_precision', 0):.4f}"
    )
    logger.info("Metric-by-metric comparison:")

    if not _is_better(new_metrics, baseline_metrics):
        logger.warning(
            "🚫 New model did NOT improve over baseline on all metrics. "
            "Skipping Hugging Face upload — keeping existing production model."
        )
        state["last_monthly_run"]  = datetime.now(timezone.utc).isoformat()
        state["next_monthly_run"]  = next_month().isoformat()
        state["monthly_run_count"] = state.get("monthly_run_count", 0) + 1
        state["last_upload_skipped"] = True
        save_schedule(state)
        return

    logger.info("✅ New model outperforms baseline! Proceeding with deployment.")

    # Step 3 — Upload to Hugging Face
    logger.info("Step 3/3 — Uploading improved model to Hugging Face Hub…")
    try:
        import toml
        secrets = toml.load(".streamlit/secrets.toml")
        hf_config = secrets.get("huggingface", {})
        hf_token  = hf_config.get("token")
        repo_id   = hf_config.get("repo_id", "unknownexplosion/SentimentABSA-v3")

        if not hf_token:
            logger.error("⚠️ Hugging Face token not found in .streamlit/secrets.toml. Skipping upload.")
        else:
            from upload_to_hub import upload_model_programmatic
            success = upload_model_programmatic(hf_token, repo_id)
            if success:
                logger.info("🚀 Monthly retraining & deployment completed successfully!")
            else:
                logger.error("⚠️ Upload task reported a failure.")
    except Exception as e:
        logger.error(f"Upload phase encountered an error: {e}", exc_info=True)

    state["last_monthly_run"] = datetime.now(timezone.utc).isoformat()
    state["next_monthly_run"] = next_month().isoformat()
    state["monthly_run_count"] = state.get("monthly_run_count", 0) + 1
    save_schedule(state)
    logger.info(f"Next scheduled monthly run: {state['next_monthly_run']}")


def print_status():
    """Show current scheduler status."""
    state = load_schedule()
    print("\n📅 Scraper & Retraining Schedule Status")
    print("─" * 45)
    last = state.get("last_run", "Never")
    nxt  = state.get("next_run") or next_sunday().isoformat()
    cnt  = state.get("run_count", 0)
    print(f"  [Weekly Scraper]")
    print(f"  Last run:   {last}")
    print(f"  Next run:   {nxt}")
    print(f"  Total runs: {cnt}\n")

    m_last = state.get("last_monthly_run", "Never")
    m_nxt  = state.get("next_monthly_run") or next_month().isoformat()
    m_cnt  = state.get("monthly_run_count", 0)
    print(f"  [Monthly Retraining]")
    print(f"  Last run:   {m_last}")
    print(f"  Next run:   {m_nxt}")
    print(f"  Total runs: {m_cnt}")

    scraped_csv = Path("outputs/scraped/reddit_reviews_all.csv")
    if scraped_csv.exists():
        import pandas as pd
        df = pd.read_csv(scraped_csv)
        print(f"\n  Total posts scraped: {len(df):,}")
        print(f"  Product breakdown:")
        for model, count in df["model"].value_counts().head(8).items():
            print(f"    • {model}: {count:,}")
    print()


def get_scheduled_runs(state: dict) -> tuple[datetime, datetime]:
    """Parse scheduled run times or compute them if not set."""
    # Weekly run
    next_week_str = state.get("next_run")
    if next_week_str:
        try:
            if next_week_str.endswith("Z"):
                next_week_str = next_week_str[:-1]
            if "+" in next_week_str:
                next_week_str = next_week_str.split("+")[0]
            nxt_week = datetime.fromisoformat(next_week_str)
        except Exception:
            nxt_week = next_sunday()
            state["next_run"] = nxt_week.isoformat()
            save_schedule(state)
    else:
        nxt_week = next_sunday()
        state["next_run"] = nxt_week.isoformat()
        save_schedule(state)
        
    # Monthly run
    next_month_str = state.get("next_monthly_run")
    if next_month_str:
        try:
            if next_month_str.endswith("Z"):
                next_month_str = next_month_str[:-1]
            if "+" in next_month_str:
                next_month_str = next_month_str.split("+")[0]
            nxt_month = datetime.fromisoformat(next_month_str)
        except Exception:
            nxt_month = next_month()
            state["next_monthly_run"] = nxt_month.isoformat()
            save_schedule(state)
    else:
        nxt_month = next_month()
        state["next_monthly_run"] = nxt_month.isoformat()
        save_schedule(state)
        
    return nxt_week, nxt_month


def run_scheduler_loop():
    """
    Infinite loop that checks if it's time to run.
    Uses persistent state to prevent infinite wait conditions or overshoot.
    """
    logger.info("🔄 Scheduler started. Press Ctrl+C to stop.")
    
    state = load_schedule()
    nxt_week, nxt_month_time = get_scheduled_runs(state)
    
    logger.info(f"   Next weekly run: {nxt_week.strftime('%A %Y-%m-%d at %H:%M')}")
    logger.info(f"   Next monthly run: {nxt_month_time.strftime('%A %Y-%m-%d at %H:%M')}")

    while True:
        now = datetime.now()
        state = load_schedule()
        nxt_week, nxt_month_time = get_scheduled_runs(state)
        
        wait_week = (nxt_week - now).total_seconds()
        wait_month = (nxt_month_time - now).total_seconds()

        executed = False
        if wait_week <= 0:
            logger.info("⏰ Time to run weekly scraping job!")
            run_weekly_job()
            executed = True
            
        if wait_month <= 0:
            logger.info("🧠 Time to run monthly model retraining job!")
            run_monthly_retraining_job()
            executed = True
            
        if executed:
            time.sleep(60)  # Wait a minute before recalculating next runs
        else:
            closest_wait = min(wait_week, wait_month)
            hours_left = closest_wait / 3600
            if hours_left > 1.0:
                logger.info(f"⏳ Next task in {hours_left:.2f} hours. Sleeping 1 hour…")
                time.sleep(3600)
            else:
                logger.info(f"⏳ Next task in {hours_left*60:.1f} minutes. Sleeping 60 seconds…")
                time.sleep(60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper and Retraining Scheduler")
    parser.add_argument("--now",    action="store_true", help="Run the weekly job immediately")
    parser.add_argument("--retrain-now", action="store_true", help="Run the monthly retraining immediately")
    parser.add_argument("--status", action="store_true", help="Show schedule status")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.now:
        run_weekly_job()
    elif args.retrain_now:
        run_monthly_retraining_job()
    else:
        run_scheduler_loop()
