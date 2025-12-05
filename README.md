
# 🌌 Antigravity: Sentiment Integration System
### Advanced Aspect-Based Sentiment Analysis & Manufacturer Feedback Loop

![Project Status](https://img.shields.io/badge/Status-Complete-success)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Abstract
**Antigravity** is an end-to-end NLP pipeline designed to ingest multi-lingual customer reviews, perform robust sentiment analysis using transformer models, and generate actionable feedback for manufacturers. By leveraging state-of-the-art Aspect-Based Sentiment Analysis (ABSA), the system moves beyond simple positive/negative binary classification to understand *why* users feel a certain way about specific product features (e.g., "camera", "battery").

The project culminates in an interactive **Streamlit Dashboard** that visualizes these insights in real-time and provides a playground for testing the model.

---

## ✨ Key Features

*   **⚡ Automated Pipeline:** Single-command execution from raw CSV to polished reports.
*   **🌍 Multi-Lingual Support:** Automatic language detection and translation (via Google Translate API) to English before processing.
*   **🧠 Deep Learning Core:**
    *   **Pre-trained BERT:** Uses `nlptown/bert-base-multilingual-uncased-sentiment` for initial scoring.
    *   **Fine-Tuned DeBERTa:** Includes a custom-trained `microsoft/deberta-v3-small` model for high-precision ABSA.
*   **🎯 Aspect-Based Analysis:** Granular extraction of sentiments towards specific features (Camera, Battery, Price, etc.).
*   **📊 Interactive Dashboard:** A modern UI built with Streamlit and Plotly for exploring trends, distributions, and raw data.
*   **📉 Automated Feedback:** Generates text-based recommendations for manufacturers based on aggregated sentiment scores.

---

## 🛠️ How it Works (Architecture)

The system follows a linear ETL (Extract, Transform, Load) architecture with a heavy emphasis on the "Transform" phase using ML models.

### 1. Data Ingestion & Preprocessing
*   **Input:** Raw CSV files containing heterogeneous review data.
*   **Cleaning:** Regex-based filtering to remove HTML, URLs, and meaningless junk characters.
*   **Translation:** Non-English reviews are detected (`langdetect`) and translated (`deep-translator`) to ensure uniform analysis.

### 2. The Sentiment Engine
The core logic relies on Transformer architecture.
*   **Sentiment Scoring:** Input text $T$ is tokenized and passed through the BERT model to obtain class probabilities $P(C|T)$ where $C \in \{1, 2, 3, 4, 5\}$ stars.
    *   $Score = \text{argmax}(P(C|T))$
    *   $Label = \text{Map}(Score) \rightarrow \{Negative, Neutral, Positive\}$

### 3. Fine-Tuned ABSA (The "Antigravity" Effect)
To achieve higher accuracy, we fine-tune a **DeBERTa** model.
*   **Dataset Generation:** We programmatically extract sentences containing target aspects (e.g., "battery") and label them using the baseline model.
*   **Training:** This creates a domain-specific dataset (`absa_training_dataset.csv`) used to supervising-ly fine-tune DeBERTa, allowing the model to learn context-specific nuances of tech reviews.

### 4. Visualization Layer
*   **Streamlit:** Serves the frontend. It dynamically loads the `outputs/` CSV artifacts.
*   **Model Integration:** The dashboard checks for the presence of the fine-tuned model on disk and prioritizes it for real-time inference in the "Playground".

---

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.8 or higher
*   Git

### Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/unknownexplosion/Sentiment-analysis.git
    cd Sentiment-analysis
    ```

2.  **Clean Setup (Optional)**
    *Remove old build artifacts to ensure a fresh start.*
    ```bash
    rm -rf .venv __pycache__ outputs/fine_tuned_absa_model
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Spacy Model**
    ```bash
    python -m spacy download en_core_web_sm
    ```

---

## 🎮 Usage

### 1. Run the Pipeline
Extract data, process sentiment, and generate reports.
```bash
python sentiment_pipeline.py
```
*   **Output:** `outputs/` folder containing CSV summaries and PNG charts.

### 2. Train the Model (Optional)
Fine-tune the DeBERTa model on your processed data.
```bash
python train_absa_model.py
```
*(Or use `ABSA_Fine_Tuning_Colab.ipynb` for free GPU training).*

### 3. Launch the Dashboard
Visualize the results in your browser.
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
Antigravity/
├── app.py                   # 🖥️ Main Streamlit Dashboard application
├── sentiment_pipeline.py    # ⚙️ Core ETL and Sentiment Analysis script
├── train_absa_model.py      # 🧠 Script for fine-tuning DeBERTa
├── create_notebook.py       # 📓 Utility to recreate the Jupyter notebook
├── requirements.txt         # 📦 Dependencies list
├── DEPLOYMENT.md            # ☁️ Cloud deployment guide
├── outputs/                 # 📂 Generated artifacts
│   ├── sentiment_output.csv         # Full labeled dataset
│   ├── absa_training_dataset.csv    # Data for fine-tuning
│   ├── manufacturer_recommendations.md
│   ├── fine_tuned_absa_model/       # (Optional) Trained model weights
│   └── plots/                       # Static PNG charts
└── final_dataset.csv        # 📄 Source data
```

---

### Phase 1: Cleanup Checklist (For Examiner)
*Before submitting, ensure the following are **REMOVED** from your archive:*
- [ ] `.venv/` or `env/` (Virtual environments)
- [ ] `__pycache__/` (Compiled Python bytecode)
- [ ] `.DS_Store` (Mac system files)
- [ ] `.ipynb_checkpoints/` (Jupyter autosaves)
- [ ] `fine_tuned_model.zip` (Original large zip file - keep the unzipped folder only if needed, or rely on script to retrain)
- [ ] `outputs/fine_tuned_absa_model/` (OPTIONAL: Remove if file size limit is tight; the code supports running without it)

---

**Author:** Anubhav Mukherjee
**Project:** Antigravity (Final Year Submission)
