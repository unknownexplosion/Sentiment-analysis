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
.rtc-wrap { font-family: 'Inter', sans-serif; color: #18181b; }

/* ── Page header ── */
.rtc-page-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #09090b;
    margin: 0 0 2px;
    line-height: 1.2;
}
.rtc-page-sub {
    font-size: 0.8rem;
    color: #71717a;
    font-weight: 400;
    margin: 0;
}

/* ── Schedule card ── */
.rtc-sched-card {
    display: flex;
    gap: 0;
    background: #fafafa;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    padding: 0;
    margin: 18px 0 0;
    overflow: hidden;
}
.rtc-sched-item {
    flex: 1;
    padding: 16px 20px;
    border-right: 1px solid #e4e4e7;
}
.rtc-sched-item:last-child { border-right: none; }
.rtc-sched-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a1a1aa;
    margin-bottom: 5px;
}
.rtc-sched-value {
    font-size: 0.85rem;
    font-weight: 600;
    color: #18181b;
    line-height: 1.3;
}
.rtc-sched-value.muted { color: #71717a; font-weight: 400; }

/* ── Status pill ── */
.rtc-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    line-height: 1;
}
.rtc-pill::before {
    content: '';
    display: inline-block;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.8;
}
.pill-idle      { background: #f4f4f5; color: #71717a; }
.pill-scraping  { background: #fef9c3; color: #854d0e; }
.pill-training  { background: #dbeafe; color: #1d4ed8; }
.pill-comparing { background: #f3e8ff; color: #6b21a8; }
.pill-success   { background: #dcfce7; color: #15803d; }
.pill-skipped   { background: #fff7ed; color: #9a3412; }
.pill-error     { background: #fee2e2; color: #991b1b; }

/* ── Section label ── */
.rtc-step-label {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 28px 0 12px;
}
.rtc-step-num {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #18181b;
    color: #fff;
    font-size: 0.65rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.rtc-step-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #52525b;
}

/* ── Metric comparison table ── */
.rtc-cmp-header {
    display: grid;
    grid-template-columns: 120px 1fr 50px 1fr;
    font-size: 0.67rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #a1a1aa;
    padding: 0 0 8px;
    border-bottom: 1px solid #e4e4e7;
}
.rtc-metric-row {
    display: grid;
    grid-template-columns: 120px 1fr 50px 1fr;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #f4f4f5;
    font-size: 0.84rem;
}
.rtc-metric-name { color: #3f3f46; font-weight: 500; }
.rtc-metric-val  {
    text-align: center;
    font-weight: 600;
    color: #09090b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.rtc-metric-arrow { text-align: center; font-size: 0.95rem; font-weight: 700; }

/* ── Progress bar ── */
.rtc-prog-wrap { margin: 6px 0 12px; }
.rtc-prog-track {
    background: #f4f4f5;
    border-radius: 9999px;
    height: 5px;
    overflow: hidden;
}
.rtc-prog-fill {
    height: 100%;
    border-radius: 9999px;
    background: #18181b;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.rtc-prog-caption {
    font-size: 0.72rem;
    color: #71717a;
    margin-top: 5px;
}

/* ── Subreddit rows ── */
.rtc-sub-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #f4f4f5;
    font-size: 0.82rem;
}
.rtc-sub-name  { color: #3f3f46; font-weight: 500; }
.rtc-sub-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #18181b;
    background: #f4f4f5;
    padding: 2px 8px;
    border-radius: 4px;
}

/* ── Mini progress bar for subreddits ── */
.rtc-sub-bar-track {
    background: #f4f4f5;
    border-radius: 9999px;
    height: 3px;
    margin: 3px 0 0;
    overflow: hidden;
}
.rtc-sub-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: #71717a;
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
.rtc-hr { border: none; border-top: 1px solid #f4f4f5; margin: 22px 0 0; }
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

# ─── Background pipeline thread ───────────────────────────────────────────────

def _run_pipeline_bg():
    """
    Runs the full retraining pipeline in a background thread so the
    Streamlit UI stays responsive.  Writes status to STATUS_FILE and logs
    to LOG_FILE — the main thread polls both.
    """
    # Clear old log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    _write_status({"phase": "scraping", "error": None, "scraped_count": 0, "sub_counts": {}})
    _log("Pipeline started.")
    _log("Step 1/4 — Scraping Reddit for Apple product reviews…")

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
            # Latest weekly file is just the one named with current week number
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

    # ── 2. Run sentiment pipeline ────────────────────────────────────────────
    _write_status({"phase": "pipeline"})
    _log("Step 2/4 — Running sentiment pipeline on scraped data…")

    if new_df is not None:
        try:
            from reddit_scraper import run_pipeline_on_new_data
            ok = run_pipeline_on_new_data(new_df)
            if ok:
                _log("✅  Sentiment pipeline complete. ABSA dataset updated.")
            else:
                _log("⚠️  Sentiment pipeline encountered issues — continuing with existing data.")
        except Exception as e:
            _log(f"⚠️  Pipeline warning: {e}  — continuing.")
    else:
        _log("   Skipping pipeline (no new data).")

    # ── 3. Train model ───────────────────────────────────────────────────────
    _write_status({"phase": "training"})
    _log("Step 3/4 — Fine-tuning DeBERTa model…")

    # Snapshot baseline before training overwrites metrics.json
    baseline = _load_metrics(METRICS_PATH)
    if baseline:
        _write_status({"baseline_metrics": baseline})
        _log(f"   Baseline  → Accuracy: {baseline.get('eval_accuracy',0):.4f}  "
             f"F1: {baseline.get('eval_f1',0):.4f}  "
             f"Precision: {baseline.get('eval_precision',0):.4f}")
        # Save a separate baseline snapshot so the UI can compare after training
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

    # ── 4. Compare & conditionally deploy ────────────────────────────────────
    _write_status({"phase": "comparing"})
    _log("Step 4/4 — Comparing new model vs baseline…")

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
        # F1 is the primary gate — must improve
        if new.get("eval_f1", 0) < old.get("eval_f1", 0):
            return False
        # At least 2 of 3 metrics (Acc, F1, Precision) must improve or hold
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
        repo_id  = secrets.get("huggingface", {}).get("repo_id", "unknownexplosion/SentimentAnalysisog")

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


def _start_pipeline():
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
    _write_status({"phase": "starting", "scraped_count": 0, "sub_counts": {},
                   "baseline_metrics": {}, "new_metrics": {}, "error": None,
                   "initial_dataset_size": initial_size})
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    t = threading.Thread(target=_run_pipeline_bg, daemon=True)
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

def _fmt_date(iso: str | None, fallback="—") -> str:
    if not iso:
        return fallback
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%a, %d %b %Y  %H:%M")
    except Exception:
        return iso[:19].replace("T", "  ")

def _metric_arrow(new_val, old_val):
    if not old_val:
        return "—", ""
    if new_val > old_val:
        return "↑", "green"
    elif new_val < old_val:
        return "↓", "red"
    return "=", "amber"


# ─── Main render function ─────────────────────────────────────────────────────

def render_retraining_center():
    st.markdown(MINIMAL_CSS, unsafe_allow_html=True)
    st.markdown('<div class="rtc-wrap">', unsafe_allow_html=True)

    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False

    status   = _read_status()
    phase    = status.get("phase", "")
    is_alive = phase not in ("", "done", "skipped", "error")

    if is_alive:
        st.session_state.pipeline_running = True

    # ── Page title ──────────────────────────────────────────────────────────
    st.markdown(
        '<p class="rtc-page-title">Model Retraining Center</p>'
        '<p class="rtc-page-sub">Reddit scrape &rarr; sentiment pipeline &rarr; DeBERTa fine-tune &rarr; champion vs challenger &rarr; deploy</p>',
        unsafe_allow_html=True,
    )

    # ── Schedule card ────────────────────────────────────────────────────────
    schedule  = _load_schedule()
    last_run  = _fmt_date(schedule.get("last_monthly_run"))
    next_run  = _fmt_date(schedule.get("next_monthly_run"))
    run_count = schedule.get("monthly_run_count", 0)
    pill_html = _pill(phase)

    st.markdown(f"""
    <div class="rtc-sched-card">
      <div class="rtc-sched-item">
        <div class="rtc-sched-label">Last Run</div>
        <div class="rtc-sched-value">{last_run}</div>
      </div>
      <div class="rtc-sched-item">
        <div class="rtc-sched-label">Next Scheduled</div>
        <div class="rtc-sched-value">{next_run}</div>
      </div>
      <div class="rtc-sched-item">
        <div class="rtc-sched-label">Status</div>
        <div style="margin-top:4px">{pill_html}</div>
      </div>
      <div class="rtc-sched-item">
        <div class="rtc-sched-label">Total Cycles</div>
        <div class="rtc-sched-value">{run_count}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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

    # ── Trigger button ────────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if not is_alive:
            if st.button("▶  Run Retraining Now", type="primary", width="stretch"):
                _start_pipeline()
                st.session_state.pipeline_running = True
                st.rerun()
        else:
            st.button("⏳  Pipeline Running…", disabled=True, width="stretch")

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

    # ── 1. Scraping panel (visible from scraping → onwards) ─────────────────
    if phase in ("scraping", "pipeline", "training", "comparing",
                 "deploying", "done", "skipped", "error"):
        _render_scraping_panel(status)

    # ── 2. Training metrics (visible from training → onwards) ────────────────
    if phase in ("training", "comparing", "deploying", "done", "skipped", "error"):
        _render_training_panel(status)

    # ── 3. Comparison panel ───────────────────────────────────────────────────
    if phase in ("comparing", "deploying", "done", "skipped", "error"):
        _render_comparison_panel(status)

    # ── 4. Terminal log (always visible once started) ─────────────────────────
    _render_log_panel()


def _render_scraping_panel(status: dict):
    st.markdown(
        '<div class="rtc-step-label">'
        '<div class="rtc-step-num">1</div>'
        '<div class="rtc-step-title">Reddit Data Scraping</div>'
        '</div>',
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


def _render_training_panel(status: dict):
    st.markdown(
        '<div class="rtc-step-label">'
        '<div class="rtc-step-num">2</div>'
        '<div class="rtc-step-title">DeBERTa Fine-tuning</div>'
        '</div>',
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


def _render_comparison_panel(status: dict):
    st.markdown(
        '<div class="rtc-step-label">'
        '<div class="rtc-step-num">3</div>'
        '<div class="rtc-step-title">Champion vs Challenger</div>'
        '</div>',
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


def _render_log_panel():
    st.markdown(
        '<div class="rtc-step-label">'
        '<div class="rtc-step-num" style="background:#52525b">4</div>'
        '<div class="rtc-step-title">Live Pipeline Log</div>'
        '</div>',
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
