
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
MODEL_CHECKPOINT = "microsoft/deberta-v3-small"  # Efficient and powerful
OUTPUT_DIR = "outputs/fine_tuned_absa_model"
DATA_PATH = "outputs/absa_training_dataset.csv"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
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

def train():
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found at {DATA_PATH}. Run sentiment_pipeline.py first.")
        return

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Filter only known labels
    df = df[df['label'].isin(['Positive', 'Negative', 'Neutral'])]
    
    # Encode labels
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['label_id'] = df['label'].map(label_map)
    
    # Split data
    train_texts = df['text'].tolist()
    train_aspects = df['aspect'].tolist()
    train_labels = df['label_id'].tolist()
    
    train_texts, val_texts, train_aspects, val_aspects, train_labels, val_labels = train_test_split(
        train_texts, train_aspects, train_labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training on {len(train_texts)} samples, validating on {len(val_texts)} samples.")

    # Tokenizer
    logger.info(f"Loading tokenizer: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Tokenize (Input Pairs: Text + Aspect)
    # This helps the model explicitly focus on the sentiment OF the aspect within the text
    train_encodings = tokenizer(train_texts, train_aspects, truncation=True, padding=True, max_length=MAX_LEN)
    val_encodings = tokenizer(val_texts, val_aspects, truncation=True, padding=True, max_length=MAX_LEN)

    train_dataset = ABSADataset(train_encodings, train_labels)
    val_dataset = ABSADataset(val_encodings, val_labels)

    # Model
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=3,
        id2label={0: 'Negative', 1: 'Neutral', 2: 'Positive'},
        label2id=label_map
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation Results: {eval_results}")

if __name__ == "__main__":
    train()
