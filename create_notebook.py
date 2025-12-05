import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# Sentiment Analysis & Feedback System

This notebook implements a complete sentiment analysis pipeline for customer reviews.
It performs:
1. Data Loading & Cleaning
2. Language Translation
3. Duplicate Removal
4. Sentiment Analysis (BERT)
5. Aggregation & Reporting
"""

code_imports = """import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from IPython.display import display, Image
from tqdm import tqdm

# Add current directory to path to import pipeline
sys.path.append(os.getcwd())

# Import functions from our pipeline script
from sentiment_pipeline import (
    load_data, preprocess_reviews, translate_and_clean, 
    handle_duplicates, analyze_sentiment, generate_absa_dataset, aggregate_model_stats, 
    generate_feedback_report, plot_results
)

# %matplotlib inline
"""

code_load = """# 1. Load Data
dataset_path = 'final_dataset.csv'
df = load_data(dataset_path)
display(df.head())
"""

code_clean = """# 2. Preprocess (Clean & Filter)
df = preprocess_reviews(df)
print(f"Rows after cleaning: {len(df)}")
"""

code_translate = """# 3. Translate & Re-clean
# This step might take time if using actual translation API
tqdm.pandas(desc="Translating & Cleaning")
df['translated_review'] = df['cleaned_review'].progress_apply(lambda x: process_row({'cleaned_review': x}))
# Note: process_row needs to be available or we rely on the pipeline function which handles it internally.
# Actually, translate_and_clean in the pipeline script already uses tqdm now.
df = translate_and_clean(df)
display(df[['original_review', 'final_review']].head())
"""

code_duplicates = """# 4. Handle Duplicates
df = handle_duplicates(df)
"""

code_sentiment = """# 5. Sentiment Analysis
df, sentiment_pipe = analyze_sentiment(df)
display(df[['model', 'final_review', 'sentiment_label', 'sentiment_score']].head())
"""

code_absa = """# 6. ABSA Dataset Generation
absa_df = generate_absa_dataset(df, sentiment_pipe)
if not absa_df.empty:
    display(absa_df.head())
else:
    print("No ABSA data generated (Spacy missing or no aspects found).")
"""

code_agg = """# 6. Aggregate Stats
stats_df = aggregate_model_stats(df)
"""

code_feedback = """# 7. Generate Feedback Report
feedback_df = generate_feedback_report(stats_df)
"""

code_display_tables = """# Display Summary Tables (Requirement 7)

print("=== Model Summary ===")
display(stats_df.head(10))

print("\\n=== Feedback Report ===")
display(feedback_df.head(10))
"""

code_save = """# 8. Save Outputs
output_dir = 'outputs'
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, 'sentiment_output.csv'), index=False)
stats_df.to_csv(os.path.join(output_dir, 'per_model_summary.csv'), index=False)
feedback_df.to_csv(os.path.join(output_dir, 'feedback_report.csv'), index=False)
if 'absa_df' in locals() and not absa_df.empty:
    absa_df.to_csv(os.path.join(output_dir, 'absa_training_dataset.csv'), index=False)

# Markdown report
md_path = os.path.join(output_dir, 'manufacturer_recommendations.md')
with open(md_path, 'w') as f:
    f.write("# Manufacturer Feedback Report\\n\\n")
    for _, row in feedback_df.iterrows():
        f.write(f"## Model: {row['model']}\\n")
        f.write(f"**Summary**: {row['summary']}\\n\\n")
        f.write(f"**Strengths**: {row['strengths']}\\n\\n")
        f.write(f"**Weaknesses**: {row['weaknesses']}\\n\\n")
        f.write(f"**Recommendations**: {row['recommendations']}\\n\\n")
        f.write("---\\n\\n")

print(f"Outputs saved to {output_dir}")
"""

code_plot = """# 9. Plots
try:
    plot_results(df, stats_df, plots_dir)
    
    # Display plots inline
    print("Global Sentiment Distribution:")
    display(Image(filename=os.path.join(plots_dir, 'global_sentiment_distribution.png')))
    
    print("Per-Model Sentiment Count:")
    display(Image(filename=os.path.join(plots_dir, 'per_model_sentiment_count.png')))
except Exception as e:
    print(f"Error plotting or displaying: {e}")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_clean),
    nbf.v4.new_code_cell(code_translate),
    nbf.v4.new_code_cell(code_duplicates),
    nbf.v4.new_code_cell(code_sentiment),
    nbf.v4.new_code_cell(code_absa),
    nbf.v4.new_code_cell(code_agg),
    nbf.v4.new_code_cell(code_feedback),
    nbf.v4.new_code_cell(code_display_tables),
    nbf.v4.new_code_cell(code_save),
    nbf.v4.new_code_cell(code_plot)
]

with open('/Users/anubhavmukherjee/Desktop/Sentiment-analysis/sentiment_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully.")
