
import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
MODEL_CHECKPOINT = "unknownexplosion/SentimentAnalysisog"
OUTPUT_DIR       = "outputs/fine_tuned_absa_model"
DATA_PATH        = "outputs/absa_training_dataset.csv"
LIVE_METRICS     = "outputs/live_training_metrics.json"   # written after every epoch
MAX_LEN          = 128
BATCH_SIZE       = 16
EPOCHS           = 8   # was 3 — increased for better convergence


# ── Live metrics callback ──────────────────────────────────────────────────────
class LiveMetricsCallback(TrainerCallback):
    """Writes current training metrics to a JSON file after every evaluation step
    so the Retraining Center dashboard can read them in real time."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        payload = {
            "epoch":        state.epoch,
            "total_epochs": args.num_train_epochs,
            "train_loss":   state.log_history[-1].get("loss", 0) if state.log_history else 0,
            "learning_rate":state.log_history[-1].get("learning_rate", 0) if state.log_history else 0,
            "eval_loss":    metrics.get("eval_loss", 0),
            "eval_accuracy":metrics.get("eval_accuracy", 0),
            "eval_f1":      metrics.get("eval_f1", 0),
            "eval_precision":metrics.get("eval_precision", 0),
        }
        try:
            os.makedirs(os.path.dirname(LIVE_METRICS), exist_ok=True)
            with open(LIVE_METRICS, "w") as f:
                json.dump(payload, f)
                
            # Append epoch completion to the UI log!
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            log_str = f"[{ts}] Epoch {state.epoch:.0f}/{args.num_train_epochs} complete → Val Acc: {payload['eval_accuracy']:.4f} | F1: {payload['eval_f1']:.4f}\n"
            with open("outputs/retraining_run.log", "a") as f:
                f.write(log_str)
                
        except Exception:
            pass


# ── Weighted loss to handle class imbalance ────────────────────────────────────
class WeightedTrainer(Trainer):
    """Overrides compute_loss to apply per-class weights (fixes 78% Positive bias)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = torch.tensor(
                self.class_weights, dtype=torch.float, device=logits.device
            )
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Dataset wrapper ────────────────────────────────────────────────────────────
class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


# ── Main training function ─────────────────────────────────────────────────────
def train():
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found at {DATA_PATH}. Run sentiment_pipeline.py first.")
        return

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Keep only valid labels and de-duplicate (text, aspect) pairs
    df = df[df['label'].isin(['Positive', 'Negative', 'Neutral'])]
    before = len(df)
    df = df.drop_duplicates(subset=['text', 'aspect'])
    logger.info(f"  Removed {before - len(df)} duplicate (text, aspect) rows. {len(df)} remain.")

    # Encode labels
    label_map     = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['label_id'] = df['label'].map(label_map)

    # ── Log class distribution & compute weights ───────────────────────────────
    dist = df['label'].value_counts()
    logger.info("Class distribution:")
    for lbl, cnt in dist.items():
        logger.info(f"  {lbl}: {cnt} ({cnt/len(df)*100:.1f}%)")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1, 2]),
        y=df['label_id'].values
    )
    logger.info(f"Class weights → Negative: {class_weights[0]:.3f}  "
                f"Neutral: {class_weights[1]:.3f}  Positive: {class_weights[2]:.3f}")

    # Train / val split
    train_texts, val_texts, train_asp, val_asp, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['aspect'].tolist(),
        df['label_id'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label_id'].tolist(),   # keep class ratios in both splits
    )
    logger.info(f"Training on {len(train_texts)} samples, validating on {len(val_texts)} samples.")

    # Tokenise (text + aspect pair — helps model focus on aspect sentiment)
    logger.info(f"Loading tokenizer: {MODEL_CHECKPOINT}")
    tokenizer      = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    train_enc      = tokenizer(train_texts, train_asp, truncation=True, padding=True, max_length=MAX_LEN)
    val_enc        = tokenizer(val_texts,   val_asp,   truncation=True, padding=True, max_length=MAX_LEN)
    train_dataset  = ABSADataset(train_enc, train_labels)
    val_dataset    = ABSADataset(val_enc,   val_labels)

    # Model
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=3,
        id2label={0: 'Negative', 1: 'Neutral', 2: 'Positive'},
        label2id=label_map,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[LiveMetricsCallback()],
    )

    logger.info("Starting training...")
    trainer.train()

    # Save model + tokeniser
    logger.info(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Evaluate & persist metrics for the dashboard
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation Results: {eval_results}")

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_results, f)
    logger.info(f"Saved metrics to {metrics_path}")

    # Clean up live metrics file when done
    if os.path.exists(LIVE_METRICS):
        os.remove(LIVE_METRICS)


if __name__ == "__main__":
    train()
