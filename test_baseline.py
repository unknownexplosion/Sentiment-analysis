import patch_transformers
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_MODEL = "unknownexplosion/SentimentABSA-v3"
DATA_PATH = "outputs/absa_training_dataset.csv"
MAX_LEN = 128
BATCH_SIZE = 16

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_baseline():
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found at {DATA_PATH}")
        return

    logger.info("Loading validation dataset split...")
    df = pd.read_csv(DATA_PATH)
    df = df[df['label'].isin(['Positive', 'Negative', 'Neutral'])]
    
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['label_id'] = df['label'].map(label_map)
    
    # We strictly use random_state=42 to perfectly match the validation set 
    # generation from the training script
    _, val_texts, _, val_aspects, _, val_labels = train_test_split(
        df['text'].tolist(), df['aspect'].tolist(), df['label_id'].tolist(), 
        test_size=0.2, random_state=42
    )
    
    logger.info(f"Loaded {len(val_texts)} validation samples.")

    logger.info(f"Downloading/Loading tokenizers and older model from Hugging Face: {HF_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL, num_labels=3, 
        id2label={0: 'Negative', 1: 'Neutral', 2: 'Positive'},
        label2id=label_map
    )

    val_encodings = tokenizer(val_texts, val_aspects, truncation=True, padding=True, max_length=MAX_LEN)
    val_dataset = ABSADataset(val_encodings, val_labels)

    # We just need evaluation, so training arguments are kept minimal
    training_args = TrainingArguments(
        output_dir="outputs/temp_eval",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Running evaluation matrices...")
    eval_results = trainer.evaluate()
    
    logger.info("="*40)
    logger.info("OLD MODEL BASELINE SCORES:")
    logger.info(f"Accuracy:  {eval_results.get('eval_accuracy')*100:.2f}%")
    logger.info(f"F1-Score:  {eval_results.get('eval_f1'):.4f}")
    logger.info(f"Precision: {eval_results.get('eval_precision'):.4f}")
    logger.info(f"Recall:    {eval_results.get('eval_recall'):.4f}")
    logger.info("="*40)
    
    # Dump this into metrics.json to instantly simulate your dashboard!
    metrics_path = "outputs/fine_tuned_absa_model/metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(eval_results, f)
    logger.info(f"Overwrote local metrics.json with baseline scores!")

if __name__ == "__main__":
    evaluate_baseline()
