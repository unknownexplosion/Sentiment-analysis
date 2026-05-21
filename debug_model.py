import json
from pathlib import Path
import traceback
from huggingface_hub import hf_hub_download
from transformers import pipeline

try:
    print("Attempting to download and patch tokenizer_config.json...")
    config_path = hf_hub_download(repo_id="unknownexplosion/SentimentABSA-v3", filename="tokenizer_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if isinstance(config.get("extra_special_tokens"), list):
        print("extra_special_tokens is a list. Patching it to an empty dictionary...")
        config["extra_special_tokens"] = {}
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Patched successfully!")
    else:
        print("extra_special_tokens is not a list. No patch needed.")
except Exception as e:
    print(f"Failed to patch tokenizer config: {e}")
    traceback.print_exc()

try:
    print("\nTesting loading unknownexplosion/SentimentABSA-v3...")
    classifier = pipeline(
        "sentiment-analysis", 
        model="unknownexplosion/SentimentABSA-v3", 
        device=-1, 
        model_kwargs={"low_cpu_mem_usage": False}
    )
    print("✅ Successfully loaded!")
except Exception as e:
    print(f"❌ Failed: {e}")
    traceback.print_exc()
