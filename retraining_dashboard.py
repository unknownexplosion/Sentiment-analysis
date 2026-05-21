"""
retraining_dashboard.py
=========================
Real-time Model Retraining Center for the Sentiment Analysis Streamlit app.

Pipeline:
  1. Scrape Reddit (real PRAW calls)
  2. Download scraped CSV
  3. Run sentiment pipeline on new data
  4. Train fine-tuned DeBERTa model
  5. Compare new model metrics vs baseline
  6. Upload to Hugging Face only if improved
"""
import patch_transformers
import streamlit as st
import pandas as pd
import json
import os
import io
import subprocess
import sys
import time
import threading
import queue
from datetime import datetime, timezone
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = Path("outputs/scraped")
SCHEDULE_FILE   = Path("outputs/scraped/schedule.json")
SCRAPED_CSV     = Path("outputs/scraped/reddit_reviews_all.csv")
WEEK_CSV_PATTERN= "outputs/scraped/"
METRICS_PATH    = Path("outputs/fine_tuned_absa_model/metrics.json")
ABSA_CSV        = Path("outputs/absa_training_dataset.csv")
LOG_FILE        = Path("outputs/retraining_run.log")
STATUS_FILE     = Path("outputs/retraining_status.json")
BASELINE_FILE   = Path("outputs/baseline_metrics.json")
STOP_FILE       = Path("outputs/pipeline_stop.flag")

SUBREDDITS = [
    "apple", "iPhone", "Mac", "iPad", "MacOS", "AppleWatch", "AirPods", "AppleMusic",
    "appleswap", "applehelp", "AppleWhatShouldIBuy", "iphonehelp", "AppleIndia",
    "iOSProgramming", "applesucks",
]

# ─── CSS ──────────────────────────────────────────────────────────────────────
MINIMAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;1,14..32,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset / base ── */
.rtc-wrap * { box-sizing: border-box; }
.rtc-wrap { font-family: 'Inter', sans-serif; color: #E4E4E7; }

/* ── Page header ── */
.rtc-page-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #F5F5F7;
    margin: 0 0 2px;
    line-height: 1.2;
}
.rtc-page-sub {
    font-size: 0.8rem;
    color: #8A8A93;
    font-weight: 400;
    margin: 0;
}

/* ── Schedule grid & cards ── */
.rtc-sched-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 20px 0;
}
@media (max-width: 768px) {
    .rtc-sched-grid {
        grid-template-columns: 1fr;
    }
}
.rtc-sched-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 24px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 220px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.rtc-sched-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.35);
    border-color: rgba(255, 255, 255, 0.15);
}
.rtc-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 12px;
}
.rtc-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 8px;
}
.rtc-card-body {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px 24px;
}
.rtc-sched-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.rtc-sched-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8A8A93;
}
.rtc-sched-value {
    font-size: 0.88rem;
    font-weight: 500;
    color: #F5F5F7;
    line-height: 1.3;
}
.rtc-sched-value.highlight {
    color: #3B82F6;
    font-weight: 600;
}
.rtc-sched-value.muted { color: #71717a; font-weight: 400; }

/* ── Status pill ── */
.rtc-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    line-height: 1;
}
.rtc-pill::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}
.pill-idle      { background: rgba(244, 244, 245, 0.08); color: #A1A1AA; }
.pill-scraping  { background: rgba(254, 249, 195, 0.08); color: #EAB308; }
.pill-training  { background: rgba(219, 234, 254, 0.08); color: #3B82F6; }
.pill-comparing { background: rgba(243, 232, 255, 0.08); color: #A855F7; }
.pill-success   { background: rgba(220, 252, 231, 0.08); color: #22C55E; }
.pill-skipped   { background: rgba(255, 247, 237, 0.08); color: #F97316; }
.pill-error     { background: rgba(254, 226, 226, 0.08); color: #EF4444; }

/* ── Section label ── */
.rtc-step-label {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 32px 0 14px;
}
.rtc-step-num {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(255, 255, 255, 0.15);
    flex-shrink: 0;
}
.rtc-step-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #A1A1AA;
}

/* ── Metric comparison table ── */
.rtc-cmp-header {
    display: grid;
    grid-template-columns: 120px 1fr 50px 1fr;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #71717A;
    padding: 0 0 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.rtc-metric-row {
    display: grid;
    grid-template-columns: 120px 1fr 50px 1fr;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 0.86rem;
}
.rtc-metric-name { color: #A1A1AA; font-weight: 500; }
.rtc-metric-val  {
    text-align: center;
    font-weight: 600;
    color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.rtc-metric-arrow { text-align: center; font-size: 0.95rem; font-weight: 700; }

/* ── Progress bar ── */
.rtc-prog-wrap { margin: 8px 0 16px; }
.rtc-prog-track {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 9999px;
    height: 6px;
    overflow: hidden;
}
.rtc-prog-fill {
    height: 100%;
    border-radius: 9999px;
    background: #3B82F6;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.rtc-prog-caption {
    font-size: 0.72rem;
    color: #8A8A93;
    margin-top: 6px;
}

/* ── Subreddit rows ── */
.rtc-sub-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 0.84rem;
}
.rtc-sub-name  { color: #A1A1AA; font-weight: 500; }
.rtc-sub-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #FFFFFF;
    background: rgba(255, 255, 255, 0.06);
    padding: 2px 8px;
    border-radius: 4px;
}

/* ── Mini progress bar for subreddits ── */
.rtc-sub-bar-track {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 9999px;
    height: 4px;
    margin: 4px 0 0;
    overflow: hidden;
}
.rtc-sub-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: #3B82F6;
}

/* ── Terminal log ── */
.rtc-terminal {
    background: #09090b;
    color: #a1a1aa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.75;
    padding: 16px 20px;
    border-radius: 10px;
    border: 1px solid #27272a;
    height: 280px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
.rtc-terminal .log-ts  { color: #52525b; }
.rtc-terminal .log-ok  { color: #4ade80; }
.rtc-terminal .log-err { color: #f87171; }
.rtc-terminal .log-warn{ color: #fbbf24; }

/* ── Divider ── */
.rtc-hr { border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 24px 0 0; }
</style>
"""

# ─── State helpers ─────────────────────────────────────────────────────────────

def _read_status() -> dict:
    """Read the live pipeline status JSON written by the background thread."""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _write_status(update: dict):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = _read_status()
        current.update(update)
        # Always stamp the heartbeat so the UI can detect dead threads
        current["heartbeat"] = datetime.now(timezone.utc).isoformat()
        with open(STATUS_FILE, "w") as f:
            json.dump(current, f, indent=2, default=str)
    except Exception:
        pass

def _load_schedule() -> dict:
    if SCHEDULE_FILE.exists():
        with open(SCHEDULE_FILE) as f:
            return json.load(f)
    return {}

def _load_metrics(path: Path) -> dict:
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _read_log() -> str:
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE) as f:
                return f.read()
    except Exception:
        pass
    return ""

def _log(msg: str):
    """Append a timestamped line to the run log file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)

# ─── Stop helpers ─────────────────────────────────────────────────────────────

def _should_stop() -> bool:
    """Check if the user requested a pipeline stop."""
    return STOP_FILE.exists()

def _stop_pipeline():
    """Signal the background thread to stop."""
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_FILE.touch()

def _clear_stop():
    """Remove the stop flag (called when starting a new run)."""
    if STOP_FILE.exists():
        STOP_FILE.unlink()

def _abort_if_stopped(phase_label: str) -> bool:
    """Check stop flag; if set, log and update status. Returns True if stopped."""
    if _should_stop():
        _log(f"🛑  Pipeline stopped by user during: {phase_label}")
        _write_status({"phase": "error", "error": "Pipeline stopped by user."})
        _clear_stop()
        return True
    return False

# ─── Background pipeline thread ───────────────────────────────────────────────

def _run_pipeline_bg(run_type="all"):
    """
    Runs scraping only, training only, or the full retraining pipeline in a background thread.
    Writes status to STATUS_FILE and logs to LOG_FILE.
    """
    # Clear old log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    _log(f"Pipeline started. Task: {run_type.upper()}")

    if run_type in ("all", "scrape"):
        _write_status({"phase": "scraping", "error": None, "scraped_count": 0, "sub_counts": {}})
        _log("Step 1/2 — Scraping Reddit for Apple product reviews…")

        # ── 1. Scrape ────────────────────────────────────────────────────────────
        try:
            from reddit_scraper import RedditScraper
            scraper = RedditScraper()
            new_df = scraper.run()

            if new_df is None or len(new_df) == 0:
                _log("⚠️  No new posts found. Using existing dataset if available.")
                new_df = None
            else:
                sub_counts = new_df["subreddit"].value_counts().to_dict()
                from datetime import datetime as _dt
                week_file = OUTPUT_DIR / f"reddit_{_dt.now().strftime('%Y-W%V')}.csv"
                _write_status({
                    "scraped_count": len(new_df),
                    "sub_counts": sub_counts,
                    "scraped_csv": str(SCRAPED_CSV),
                    "week_csv": str(week_file),
                })
                _log(f"✅  Scraped {len(new_df)} posts across {new_df['subreddit'].nunique()} subreddits.")
                for sub, cnt in sub_counts.items():
                    _log(f"   r/{sub}: {cnt} posts")

        except Exception as e:
            _log(f"❌  Scraping failed: {e}")
            _write_status({"phase": "error", "error": str(e)})
            return

        # ── Check stop before step 2 ──
        if _abort_if_stopped("Scraping"): return

        # ── 2. Run sentiment pipeline ────────────────────────────────────────────
        _write_status({"phase": "pipeline"})
        step_prefix = "Step 2/2" if run_type == "scrape" else "Step 2/4"
        _log(f"{step_prefix}a — Running production sentiment pipeline (SentimentABSA-v3)…")

        if new_df is not None:
            try:
                from reddit_scraper import run_pipeline_on_new_data, run_retraining_pipeline_on_new_data
                ok_prod = run_pipeline_on_new_data(new_df)
                if ok_prod:
                    _log("✅  Production sentiment pipeline complete (sentiment_output.csv updated).")
                else:
                    _log("⚠️  Production sentiment pipeline encountered issues.")

                _log(f"{step_prefix}b — Labeling data for retraining (nlptown teacher model)…")
                ok_train = run_retraining_pipeline_on_new_data(new_df)
                if ok_train:
                    _log("✅  Retraining labeling pipeline complete (absa_training_dataset.csv updated).")
                else:
                    _log("⚠️  Retraining labeling pipeline encountered issues.")
            except Exception as e:
                _log(f"⚠️  Pipeline warning: {e}  — continuing.")
        else:
            _log("   Skipping pipelines (no new data).")

        # Update scraper schedule stats
        from weekly_scheduler import load_schedule, save_schedule, next_sunday
        state = load_schedule()
        state["last_run"]  = datetime.now(timezone.utc).isoformat()
        state["next_run"]  = next_sunday().isoformat()
        state["run_count"] = state.get("run_count", 0) + 1
        save_schedule(state)

        if run_type == "scrape":
            _write_status({"phase": "done"})
            _log("✅  Scraping and sentiment pipeline complete.")
            return

    # ── Check stop before step 3 ──
    if _abort_if_stopped("Scraping / Sentiment Pipeline"): return

    if run_type in ("all", "train"):
        _write_status({"phase": "training"})
        _log("Step 1/2 — Fine-tuning DeBERTa model…") if run_type == "train" else _log("Step 3/4 — Fine-tuning DeBERTa model…")

        # Snapshot baseline before training overwrites metrics.json
        baseline = _load_metrics(METRICS_PATH)
        if baseline:
            _write_status({"baseline_metrics": baseline})
            _log(f"   Baseline  → Accuracy: {baseline.get('eval_accuracy',0):.4f}  "
                 f"F1: {baseline.get('eval_f1',0):.4f}  "
                 f"Precision: {baseline.get('eval_precision',0):.4f}")
            with open(BASELINE_FILE, "w") as bf:
                json.dump(baseline, bf)
        else:
            _log("   No baseline metrics found — will deploy unconditionally if training succeeds.")

        try:
            from train_absa_model import train
            train()
            _log("✅  Training complete.")
        except Exception as e:
            _log(f"❌  Training failed: {e}")
            _write_status({"phase": "error", "error": str(e)})
            return

        # ── Check stop before step 4 ──
        if _abort_if_stopped("Training"): return

        # ── 4. Compare & conditionally deploy ────────────────────────────────────
        _write_status({"phase": "comparing"})
        _log("Step 2/2 — Comparing new model vs baseline…") if run_type == "train" else _log("Step 4/4 — Comparing new model vs baseline…")

        new_metrics = _load_metrics(METRICS_PATH)
        if not new_metrics:
            _log("❌  Could not read new metrics.json. Aborting deploy.")
            _write_status({"phase": "error", "error": "metrics.json not found after training"})
            return

        _write_status({"new_metrics": new_metrics})
        _log(f"   New model → Accuracy: {new_metrics.get('eval_accuracy',0):.4f}  "
             f"F1: {new_metrics.get('eval_f1',0):.4f}  "
             f"Precision: {new_metrics.get('eval_precision',0):.4f}")

        def _is_better(new, old) -> bool:
            if not old:
                return True
            if new.get("eval_f1", 0) < old.get("eval_f1", 0):
                return False
            keys = ["eval_accuracy", "eval_f1", "eval_precision"]
            improved_count = sum(1 for k in keys if new.get(k, 0) >= old.get(k, 0))
            return improved_count >= 2

        improved = _is_better(new_metrics, baseline)

        if not improved:
            _log("🚫  New model F1 did not improve over baseline. Deployment skipped.")
            _log("   Production model remains unchanged.")
            _write_status({"phase": "skipped", "new_metrics": new_metrics})
            _update_schedule(deployed=False)
            return

        _log("🎉  New model outperforms baseline! Uploading to Hugging Face…")
        _write_status({"phase": "deploying"})

        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            hf_token = secrets.get("huggingface", {}).get("token")
            repo_id  = secrets.get("huggingface", {}).get("repo_id", "unknownexplosion/SentimentABSA-v3")

            if hf_token:
                from upload_to_hub import upload_model_programmatic
                ok = upload_model_programmatic(hf_token, repo_id)
                if ok:
                    _log(f"🚀  Model deployed to {repo_id}.")
                else:
                    _log("⚠️  Upload reported failure — check HF token/connectivity.")
            else:
                _log("⚠️  No HF token in secrets.toml — skipping upload.")
        except Exception as e:
            _log(f"⚠️  Deploy warning: {e}")

        _write_status({"phase": "done"})
        _update_schedule(deployed=True)
        _log("✅  Pipeline complete.")


def _update_schedule(deployed: bool):
    from weekly_scheduler import load_schedule, save_schedule, next_month
    state = load_schedule()
    state["last_monthly_run"]  = datetime.now(timezone.utc).isoformat()
    state["next_monthly_run"]  = next_month().isoformat()
    state["monthly_run_count"] = state.get("monthly_run_count", 0) + 1
    state["last_upload_skipped"] = not deployed
    save_schedule(state)


def _start_pipeline(run_type="all"):
    """Launch the pipeline on a daemon thread and reset state/log files."""
    try:
        import pandas as pd
        _tmp_init = pd.read_csv(ABSA_CSV, usecols=['label', 'text', 'aspect'])
        _tmp_init = _tmp_init[_tmp_init['label'].isin(['Positive', 'Negative', 'Neutral'])]
        _tmp_init = _tmp_init.drop_duplicates(subset=['text', 'aspect'])
        initial_size = len(_tmp_init)
    except Exception:
        initial_size = 0

    # Wipe old state
    _write_status({"phase": "starting", "run_type": run_type, "scraped_count": 0, "sub_counts": {},
                   "baseline_metrics": {}, "new_metrics": {}, "error": None,
                   "initial_dataset_size": initial_size})
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    _clear_stop()   # ensure no leftover stop flag

    t = threading.Thread(target=lambda: _run_pipeline_bg(run_type), daemon=True)
    t.start()


# ─── UI helpers ───────────────────────────────────────────────────────────────

_PHASE_LABELS = {
    "": "Idle",
    "starting":  "Starting",
    "scraping":  "Scraping Reddit",
    "pipeline":  "Processing Data",
    "training":  "Training Model",
    "comparing": "Comparing Metrics",
    "deploying": "Deploying",
    "done":      "Completed",
    "skipped":   "Model Not Updated",
    "error":     "Error",
}

_PHASE_PILL = {
    "": "idle",
    "starting":  "scraping",
    "scraping":  "scraping",
    "pipeline":  "scraping",
    "training":  "training",
    "comparing": "comparing",
    "deploying": "training",
    "done":      "success",
    "skipped":   "skipped",
    "error":     "error",
}

def _pill(phase: str) -> str:
    label = _PHASE_LABELS.get(phase, phase.title())
    cls   = _PHASE_PILL.get(phase, "idle")
    return f'<span class="rtc-pill pill-{cls}">{label}</span>'

def to_naive_datetime(val) -> datetime | None:
    if not val:
        return None
    if isinstance(val, datetime):
        return val.replace(tzinfo=None)
    try:
        s = str(val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None)
    except Exception:
        return None

def _fmt_date(iso: str | None, fallback="—") -> str:
    dt = to_naive_datetime(iso)
    if not dt:
        return fallback
    return dt.strftime("%a, %d %b %Y %H:%M")


def _metric_arrow(new_val, old_val):
    if not old_val:
        return "—", ""
    if new_val > old_val:
        return "↑", "green"
    elif new_val < old_val:
        return "↓", "red"
    return "=", "amber"


# ─── Stale-thread detection ────────────────────────────────────────────────────
_HEARTBEAT_TIMEOUT_SEC = 300   # 5 minutes without a heartbeat → thread is dead

def _check_stale_pipeline(status: dict) -> dict:
    """If the pipeline claims to be running but the heartbeat is stale,
    mark it as errored so the UI unlocks for the user."""
    phase = status.get("phase", "")
    if phase in ("", "done", "skipped", "error"):
        return status

    hb = status.get("heartbeat")
    if not hb:
        # Legacy status without heartbeat — trust it for now
        return status

    try:
        hb_dt = datetime.fromisoformat(hb.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - hb_dt).total_seconds()
        if age > _HEARTBEAT_TIMEOUT_SEC:
            _log(f"⚠️  Pipeline thread heartbeat stale ({age:.0f}s). Marking as error.")
            status["phase"] = "error"
            status["error"] = (
                f"Pipeline thread stopped responding (no heartbeat for {age/60:.0f} min). "
                f"Last known phase: {phase}. Please re-run."
            )
            _write_status(status)
    except Exception:
        pass
    return status


# ─── Main render function ─────────────────────────────────────────────────────

def render_retraining_center():
    st.markdown(MINIMAL_CSS, unsafe_allow_html=True)
    st.markdown('<div class="rtc-wrap">', unsafe_allow_html=True)

    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False

    status   = _read_status()
    status   = _check_stale_pipeline(status)   # auto-recover dead threads
    phase    = status.get("phase", "")
    is_alive = phase not in ("", "done", "skipped", "error")
    active_run_type = status.get("run_type", "all")

    if is_alive:
        st.session_state.pipeline_running = True

    # ── Page title ──────────────────────────────────────────────────────────
    st.markdown(
        '<p class="rtc-page-title">Model Retraining Center</p>'
        '<p class="rtc-page-sub">Reddit scrape &rarr; sentiment pipeline &rarr; DeBERTa fine-tune &rarr; champion vs challenger &rarr; deploy</p>',
        unsafe_allow_html=True,
    )

    # ── Load and self-heal schedules timezone-safely ──────────────────────────
    from weekly_scheduler import load_schedule, save_schedule, next_sunday, next_month
    
    schedule  = load_schedule()
    modified = False
    
    # 1. Weekly Scraper schedule dates
    last_scrape_raw = schedule.get("last_run")
    next_scrape_raw = schedule.get("next_run")
    
    dt_next_scrape = to_naive_datetime(next_scrape_raw)
    if not dt_next_scrape or dt_next_scrape < datetime.now():
        dt_next_scrape = next_sunday()
        schedule["next_run"] = dt_next_scrape.isoformat()
        modified = True
        
    last_scrape_str = _fmt_date(last_scrape_raw)
    next_scrape_str = _fmt_date(dt_next_scrape.isoformat())
    scrape_count = schedule.get("run_count", 0)
    
    # 2. Monthly Retrainer schedule dates
    last_train_raw = schedule.get("last_monthly_run")
    next_train_raw = schedule.get("next_monthly_run")
    
    dt_next_train = to_naive_datetime(next_train_raw)
    if not dt_next_train or dt_next_train < datetime.now():
        dt_next_train = next_month()
        schedule["next_monthly_run"] = dt_next_train.isoformat()
        modified = True
        
    last_train_str = _fmt_date(last_train_raw)
    next_train_str = _fmt_date(dt_next_train.isoformat())
    train_count = schedule.get("monthly_run_count", 0)
    
    if modified:
        save_schedule(schedule)

    # Determine scraper pill and retrainer pill based on active phase and run_type
    scraper_pill_html = _pill("")  # Idle
    retrainer_pill_html = _pill("")  # Idle
    
    if phase == "error":
        err_pill = _pill("error")
        if active_run_type == "scrape":
            scraper_pill_html = err_pill
        elif active_run_type == "train":
            retrainer_pill_html = err_pill
        else:
            scraper_pill_html = err_pill
            retrainer_pill_html = err_pill
    elif phase in ("done", "skipped"):
        success_pill = _pill("done" if phase == "done" else "skipped")
        if active_run_type == "scrape":
            scraper_pill_html = success_pill
        elif active_run_type == "train":
            retrainer_pill_html = success_pill
        else:
            scraper_pill_html = _pill("done")
            retrainer_pill_html = success_pill
    elif is_alive:
        if active_run_type == "scrape":
            scraper_pill_html = _pill(phase)
        elif active_run_type == "train":
            retrainer_pill_html = _pill(phase)
        else:
            # "all"
            if phase in ("starting", "scraping", "pipeline"):
                scraper_pill_html = _pill(phase)
                retrainer_pill_html = '<span class="rtc-pill pill-idle">Waiting</span>'
            else:
                scraper_pill_html = _pill("done")
                retrainer_pill_html = _pill(phase)

    # ── Render Side-by-Side Glassmorphic Grid ──
    st.markdown('<div class="rtc-sched-grid">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="rtc-sched-card">
          <div>
            <div class="rtc-card-header">
              <span class="rtc-card-title">📥 Weekly Reddit Scraper</span>
              <div>{scraper_pill_html}</div>
            </div>
            <div class="rtc-card-body">
              <div class="rtc-sched-item">
                <div class="rtc-sched-label">Last Run</div>
                <div class="rtc-sched-value">{last_scrape_str}</div>
              </div>
              <div class="rtc-sched-item">
                <div class="rtc-sched-label">Next Scheduled</div>
                <div class="rtc-sched-value highlight">{next_scrape_str}</div>
              </div>
              <div class="rtc-sched-item">
                <div class="rtc-sched-label">Total Cycles</div>
                <div class="rtc-sched-value">{scrape_count}</div>
              </div>
              <div class="rtc-sched-item">
                <div class="rtc-sched-label">Frequency</div>
                <div class="rtc-sched-value">Weekly (Sundays)</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if not is_alive:
            if st.button("📥  Trigger Scraper & Pipeline", type="primary", width="stretch"):
                _start_pipeline(run_type="scrape")
                st.session_state.pipeline_running = True
                st.rerun()
        else:
            if active_run_type == "scrape":
                st.button("⏳ Scraper Running...", disabled=True, width="stretch")
            else:
                st.button("📥 Trigger Scraper", disabled=True, width="stretch")
                
    with col2:
        st.markdown(f"""
        <div class="rtc-sched-card">
          <div>
            <div class="rtc-card-header">
              <span class="rtc-card-title">🧠 Model Fine-Tuning & Deploy</span>
              <div>{retrainer_pill_html}</div>
            </div>
            <div class="rtc-card-body">
              <div class="rtc-sched-item">
                <div class="rtc-sched-label">Last Run</div>
                <div class="rtc-sched-value">{last_train_str}</div>
              </div>
              <div class="rtc-sched-item">
                <div class="rtc-sched-label">Next Scheduled</div>
                <div class="rtc-sched-value highlight">{next_train_str}</div>
              </div>
              <div class="rtc-sched-item">
                <div class="rtc-sched-label">Total Cycles</div>
                <div class="rtc-sched-value">{train_count}</div>
              </div>
              <div class="rtc-sched-item">
                <div class="rtc-sched-label">Frequency</div>
                <div class="rtc-sched-value">Monthly (1st)</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if not is_alive:
            if st.button("🧠  Trigger Model Retraining", type="primary", width="stretch"):
                _start_pipeline(run_type="train")
                st.session_state.pipeline_running = True
                st.rerun()
        else:
            if active_run_type == "train":
                st.button("⏳ Retraining Running...", disabled=True, width="stretch")
            else:
                st.button("🧠 Trigger Retraining", disabled=True, width="stretch")
                
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Training Data Pool card (always-visible, live from CSV) ──────────────
    try:
        import pandas as pd
        _pool_df = pd.read_csv(ABSA_CSV, usecols=['label', 'text', 'aspect'])
        raw_rows     = len(_pool_df)
        _pool_df     = _pool_df[_pool_df['label'].isin(['Positive', 'Negative', 'Neutral'])]
        valid_rows   = len(_pool_df)
        _pool_df     = _pool_df.drop_duplicates(subset=['text', 'aspect'])
        deduped_rows = len(_pool_df)
        dupes_removed = valid_rows - deduped_rows
        train_rows   = int(deduped_rows * 0.8)
        val_rows     = deduped_rows - train_rows
        dist         = _pool_df['label'].value_counts().to_dict()
        pos_pct      = dist.get('Positive', 0) / deduped_rows * 100
        neg_pct      = dist.get('Negative', 0) / deduped_rows * 100
        neu_pct      = dist.get('Neutral', 0)  / deduped_rows * 100
        pool_ok      = True
    except Exception:
        pool_ok = False

    if pool_ok:
        st.markdown("**📊 Live Training Data Pool** — `absa_training_dataset.csv`")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Raw Rows",           f"{raw_rows:,}")
        c2.metric("Dupes Removed",      f"{dupes_removed:,}")
        c3.metric("Valid for Training", f"{deduped_rows:,}", help="Unique clean rows going into the model after deduplication")
        c4.metric("Study Split (80%)",  f"{train_rows:,}",  help="Rows the AI actively learns from")
        c5.metric("Exam Split (20%)",   f"{val_rows:,}",    help="Hidden rows used only for final F1 evaluation")
        c6.metric("Label Balance",      f"P:{pos_pct:.0f}% N:{neg_pct:.0f}% Nu:{neu_pct:.0f}%")
    else:
        st.warning("Training dataset not found at `outputs/absa_training_dataset.csv`. Run a scrape first.")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── Universal Stop Button (only visible when running) ──────────────────────
    if is_alive:
        col_stop, _ = st.columns([1, 1])
        with col_stop:
            if st.button("⏹  Stop Pipeline", type="secondary", width="stretch"):
                _stop_pipeline()
                _log("🛑  Stop requested by user. Pipeline will halt after current step.")
                st.toast("Stop signal sent — pipeline will halt after the current step.", icon="🛑")
                st.rerun()
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── Auto-refresh while running ────────────────────────────────────────────
    if is_alive:
        _render_live_status(status)
        time.sleep(2)
        st.rerun()
    elif phase in ("done", "skipped", "error"):
        _render_live_status(status)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Live status panels ───────────────────────────────────────────────────────

def _render_live_status(status: dict):
    phase = status.get("phase", "")
    active_run_type = status.get("run_type", "all")

    step_num = 1

    # ── 1. Scraping panel (visible from scraping → onwards for scrape/all tasks) ──
    if active_run_type in ("scrape", "all"):
        if phase in ("scraping", "pipeline", "training", "comparing",
                     "deploying", "done", "skipped", "error"):
            _render_scraping_panel(status, step_num)
            step_num += 1

    # ── 2. Training metrics (visible from training → onwards for train/all tasks) ──
    if active_run_type in ("train", "all"):
        if phase in ("training", "comparing", "deploying", "done", "skipped", "error"):
            _render_training_panel(status, step_num)
            step_num += 1

    # ── 3. Comparison panel (visible from comparing → onwards for train/all tasks) ──
    if active_run_type in ("train", "all"):
        if phase in ("comparing", "deploying", "done", "skipped", "error"):
            _render_comparison_panel(status, step_num)
            step_num += 1

    # ── 4. Terminal log (always visible once started) ─────────────────────────
    _render_log_panel(step_num)


def _render_scraping_panel(status: dict, step_num: int):
    st.markdown(
        f'<div class="rtc-step-label">'
        f'<div class="rtc-step-num">{step_num}</div>'
        f'<div class="rtc-step-title">Reddit Data Scraping</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    scraped    = status.get("scraped_count", 0)
    sub_counts = status.get("sub_counts", {})
    phase      = status.get("phase", "")

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Posts Scraped",  f"{scraped:,}")
    kpi_cols[1].metric("Subreddits Hit", str(len(sub_counts)) if sub_counts else "—")
    kpi_cols[2].metric("Phase",          _PHASE_LABELS.get(phase, phase.title()))

    if sub_counts:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        total = max(scraped, 1)
        for sub, cnt in sorted(sub_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            st.markdown(
                f'<div class="rtc-sub-row">'
                f'<span class="rtc-sub-name">r/{sub}</span>'
                f'<span class="rtc-sub-count">{cnt:,}</span>'
                f'</div>'
                f'<div class="rtc-sub-bar-track"><div class="rtc-sub-bar-fill" style="width:{pct:.1f}%"></div></div>',
                unsafe_allow_html=True,
            )

    # Download button
    week_csv = Path(status.get("week_csv", "")) if status.get("week_csv") else None
    if week_csv and week_csv.exists():
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        with open(week_csv, "rb") as f:
            csv_bytes = f.read()
        st.download_button(
            label="↓  Download Scraped CSV",
            data=csv_bytes,
            file_name=week_csv.name,
            mime="text/csv",
        )
        try:
            preview_df = pd.read_csv(week_csv, nrows=10)
            with st.expander("Preview — first 10 rows"):
                st.dataframe(preview_df, width="stretch", hide_index=True)
        except Exception:
            pass

    if phase == "error" and status.get("run_type") == "scrape":
        err = status.get("error", "Unknown error")
        st.error(f"❌ **Scraping Error** — {err}")


def _render_training_panel(status: dict, step_num: int):
    st.markdown(
        f'<div class="rtc-step-label">'
        f'<div class="rtc-step-num">{step_num}</div>'
        f'<div class="rtc-step-title">DeBERTa Fine-tuning</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    LIVE_METRICS = Path("outputs/live_training_metrics.json")
    live_exists  = LIVE_METRICS.exists()
    live         = _load_metrics(LIVE_METRICS) if live_exists else {}

    epoch      = int(live.get("epoch", 0))
    acc        = live.get("eval_accuracy", 0) * 100
    ev_loss    = live.get("eval_loss", 0)
    train_loss = live.get("train_loss", 0)
    lr         = live.get("learning_rate", 0)

    try:
        from train_absa_model import EPOCHS as TOTAL_EPOCHS
    except Exception:
        TOTAL_EPOCHS = 8

    phase = status.get("phase", "")
    pct   = min(100, (epoch / TOTAL_EPOCHS) * 100) if TOTAL_EPOCHS else 0

    try:
        import pandas as pd
        _tmp_df = pd.read_csv("outputs/absa_training_dataset.csv", usecols=['label', 'text', 'aspect'])
        _tmp_df = _tmp_df[_tmp_df['label'].isin(['Positive', 'Negative', 'Neutral'])]
        _tmp_df = _tmp_df.drop_duplicates(subset=['text', 'aspect'])
        current_len = len(_tmp_df)
        data_count  = f"{current_len:,}"
        initial     = status.get("initial_dataset_size", current_len)
        delta       = current_len - initial
        delta_str   = f"{delta:,} new" if delta > 0 else None
    except Exception:
        data_count = "—"
        delta_str  = None

    # Progress bar
    st.markdown(
        f'<div class="rtc-prog-wrap">'
        f'<div class="rtc-prog-track" style="background:#f4f4f5; border-radius:9999px; height:6px; overflow:hidden;">'
        f'<div style="width:{pct:.0f}%; height:100%; border-radius:9999px; background:#18181b; transition:width 0.6s ease;"></div>'
        f'</div>'
        f'<div class="rtc-prog-caption" style="font-size:0.72rem; color:#71717a; margin-top:5px;">'
        f'Epoch {epoch} of {TOTAL_EPOCHS} &nbsp;&middot;&nbsp; {pct:.0f}% complete'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # If model is training but no epoch has finished yet, show status
    if phase == "training" and not live_exists:
        st.info("⏳ **Training in progress...** Waiting for Epoch 1 to complete before metrics appear. This may take several minutes on CPU.")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Data Pool",     data_count, delta_str)
    m2.metric("Epoch",         f"{epoch} / {TOTAL_EPOCHS}")
    m3.metric("Val Accuracy",  f"{acc:.2f}%"      if acc        else "—")
    m4.metric("Train Loss",    f"{train_loss:.4f}" if train_loss else "—")
    m5.metric("Val Loss",      f"{ev_loss:.4f}"    if ev_loss   else "—")
    m6.metric("Learning Rate", f"{lr:.2e}"         if lr        else "—")


def _render_comparison_panel(status: dict, step_num: int):
    st.markdown(
        f'<div class="rtc-step-label">'
        f'<div class="rtc-step-num">{step_num}</div>'
        f'<div class="rtc-step-title">Champion vs Challenger</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    phase    = status.get("phase", "")
    baseline = _load_metrics(BASELINE_FILE) or status.get("baseline_metrics", {})
    new_m    = _load_metrics(METRICS_PATH)  or status.get("new_metrics", {})

    metrics_to_show = [
        ("eval_accuracy",  "Accuracy",  lambda v: f"{v*100:.2f}%"),
        ("eval_f1",        "F1",        lambda v: f"{v:.4f}"),
        ("eval_precision", "Precision", lambda v: f"{v:.4f}"),
        ("eval_loss",      "Val Loss",  lambda v: f"{v:.4f}"),
    ]

    st.markdown(
        '<div class="rtc-cmp-header">'
        '<span>Metric</span><span style="text-align:center">Baseline</span><span></span><span style="text-align:center">New Model</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for key, label, fmt in metrics_to_show:
        old_v = baseline.get(key, 0)
        new_v = new_m.get(key, 0)
        arrow, color = _metric_arrow(new_v, old_v)
        if key == "eval_loss":
            arrow, color = _metric_arrow(old_v, new_v)

        old_str = fmt(old_v) if old_v else "—"
        new_str = fmt(new_v) if new_v else "—"
        hex_col = "#16a34a" if color == "green" else "#dc2626" if color == "red" else "#d97706"

        st.markdown(
            f'<div class="rtc-metric-row">'
            f'<span class="rtc-metric-name">{label}</span>'
            f'<span class="rtc-metric-val">{old_str}</span>'
            f'<span class="rtc-metric-arrow" style="color:{hex_col}">{arrow}</span>'
            f'<span class="rtc-metric-val">{new_str}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if phase == "done":
        st.success("✅ **Model Updated** — New model improved on F1 (primary) and outperforms baseline. Deployed to Hugging Face.")
    elif phase == "skipped":
        st.warning("🚫 **Model NOT Updated** — New model F1 did not improve over the production baseline. Existing production model remains unchanged.")
    elif phase == "error":
        err = status.get("error", "Unknown error")
        st.error(f"❌ **Pipeline Error** — {err}")


def _render_log_panel(step_num: int):
    st.markdown(
        f'<div class="rtc-step-label">'
        f'<div class="rtc-step-num" style="background:#52525b">{step_num}</div>'
        f'<div class="rtc-step-title">Live Pipeline Log</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    import html as _html
    log_text = _read_log() or "Waiting for pipeline to start\u2026"
    lines    = log_text.strip().split("\n")[-80:]
    # Escape every line so emojis/arrows never break the HTML renderer
    escaped  = "\n".join(_html.escape(ln) for ln in lines)
    st.markdown(
        f'<div style="'
        f'background:#09090b; color:#a1a1aa; '
        f'font-family: JetBrains Mono, Consolas, monospace; font-size:0.75rem; '
        f'line-height:1.75; padding:16px 20px; border-radius:10px; '
        f'border:1px solid #27272a; height:280px; overflow-y:auto; '
        f'white-space:pre-wrap; word-break:break-word;'
        f'">{escaped}</div>',
        unsafe_allow_html=True,
    )
