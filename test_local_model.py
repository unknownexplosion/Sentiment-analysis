from transformers import pipeline
import os

try:
    print("Testing model loading...")
    base_dir = os.getcwd()
    model_path = os.path.join(base_dir, "outputs", "fine_tuned_absa_model")
    
    # This matches the EXACT call in app.py
    classifier = pipeline(
        "sentiment-analysis", 
        model=model_path, 
        device=-1, 
        model_kwargs={"low_cpu_mem_usage": False}
    )
    
    print("✅ Model loaded successfully!")
    
    print("Testing inference...")
    result = classifier("The battery is terrible but screen is good.")
    print(f"✅ Inference successful: {result}")

except Exception as e:
    print(f"❌ Error: {e}")
