# Sentiment Analysis & Feedback System

This project implements a complete sentiment analysis pipeline for customer reviews. It automatically cleans text, translates non-English reviews, handles duplicates, and performs sentiment analysis using a BERT-based model. It generates detailed feedback reports and visualizations for manufacturers.

## Features
- **Automated Data Cleaning**: Removes noise, URLs, HTML, and meaningless content.
- **Multi-language Support**: Detects and translates reviews to English.
- **Sentiment Analysis**: Uses `nlptown/bert-base-multilingual-uncased-sentiment` for accurate 1-5 star rating analysis.
- **Strategic Reporting**: Generates actionable insights, strengths, weaknesses, and recommendations.
- **Visualizations**: Produces sentiment distribution charts.

## Project Structure
- `sentiment_pipeline.py`: The main script that runs the analysis.
- `sentiment_analysis.ipynb`: Jupyter Notebook for interactive exploration.
- `requirements.txt`: List of dependencies.
- `final_dataset.csv`: Input dataset (place your file here).
- `outputs/`: Generated reports and plots.

## Setup & Installation

1. **Prerequisites**: Ensure you have Python 3.8+ installed.

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This installs `torch` and `transformers` which are required for the sentiment model.*

## How to Run

### Option 1: Run the Pipeline Script
To generate all reports and plots automatically:
```bash
python3 sentiment_pipeline.py
```
Outputs will be saved in the `outputs/` folder.

### Option 2: Interactive Notebook
To view the analysis step-by-step with visualizations:
```bash
jupyter notebook sentiment_analysis.ipynb
```

## Output Files
- `sentiment_output.csv`: Detailed row-by-row sentiment scores.
- `per_model_summary.csv`: Aggregated statistics for each model.
- `feedback_report.csv`: High-level feedback and recommendations.
- `manufacturer_recommendations.md`: Formatted report for stakeholders.
