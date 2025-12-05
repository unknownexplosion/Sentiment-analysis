import subprocess
import os
import sys
import time

def run_command(command, description):
    """Runs a shell command and checks for success."""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {description}...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        # Stream output to console
        process = subprocess.run(command, shell=True, check=True)
        elapsed = time.time() - start_time
        print(f"\n✅ COMPLETED: {description} in {elapsed:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description}")
        print(f"Error Code: {e.returncode}")
        return False

def main():
    print("""
    🍏 Apple Sentiment Analysis: End-to-End Orchestrator
    ----------------------------------------------------
    This script will:
    1. Run the Sentiment Pipeline (Data Cleaning & labeling)
    2. Check for the generated ABSA Dataset
    3. Fine-Tune the DeBERTa model (if data exists)
    """)
    
    # Step 1: Run Pipeline
    pipeline_success = run_command("python sentiment_pipeline.py", "Sentiment Analysis Pipeline")
    if not pipeline_success:
        print("created pipeline failed. Aborting.")
        sys.exit(1)
        
    # Step 2: Check for Dataset
    dataset_path = "outputs/absa_training_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"\n⚠️ WARNING: {dataset_path} was not found.")
        print("Skipping Model Training.")
        sys.exit(0)
        
    print(f"\n📂 Found ABSA Dataset: {dataset_path}")
    
    # Step 3: Run Training
    print("\n⚠️ NOTE: Training might take some time (approx 5-10 mins on CPU, faster on GPU).")
    training_success = run_command("python train_absa_model.py", "DeBERTa Model Fine-Tuning")
    
    if training_success:
        print("\n🎉 Model Training Successful! Now generating Final Report...")
        
        # Step 4: Run Pipeline Again (For Final Report)
        final_pipeline_success = run_command("python sentiment_pipeline.py", "Final Analysis (Using New Model)")
        if final_pipeline_success:
             print("\n🚀 SYSTEM COMPLETE!")
             print(f"   - Final Report: outputs/manufacturer_recommendations.md")
             print("\n👉 You can now run the dashboard: streamlit run app.py")
        else:
             print("❌ Final Analysis failed.")
    else:
        print("\n❌ Model Training failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
