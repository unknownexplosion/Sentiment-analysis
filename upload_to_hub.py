import os
import sys
try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("❌ Error: 'huggingface_hub' is not installed.")
    print("👉 Install it via: pip install huggingface_hub")
    sys.exit(1)

def upload_model():
    print("🚀 Hugging Face Model Uploader")
    print("------------------------------")
    print("This script will upload your fine-tuned model to the Hugging Face Hub.")
    print("This is required for Streamlit Cloud deployment (files > 100MB).\n")

    # 1. Login
    print("Step 1: Authentication")
    print("If you are not logged in, you will be prompted for your token.")
    print("Get your token here: https://huggingface.co/settings/tokens (Write access)")
    login()

    # 2. Configuration
    print("\nStep 2: Repository Config")
    user_input = input("Enter your HF Username (e.g., 'johndoe'): ").strip()
    model_name = input("Enter a name for your model repo (e.g., 'apple-absa-v1'): ").strip()
    
    if not user_input or not model_name:
        print("❌ Username and Model Name are required.")
        return

    repo_id = f"{user_input}/{model_name}"
    local_folder = "outputs/fine_tuned_absa_model"
    
    if not os.path.exists(local_folder):
        print(f"❌ Error: Model folder '{local_folder}' not found.")
        print("   Did you run 'run_full_system.py'?")
        return

    # 3. Upload
    print(f"\nStep 3: Uploading to {repo_id}...")
    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True)
        
        api.upload_folder(
            folder_path=local_folder,
            repo_id=repo_id,
            repo_type="model"
        )
        print("\n✅ SUCCESS! Model uploaded.")
        print("---------------------------------------------------")
        print(f"Start using it in your app by setting:")
        print(f"HF_MODEL_ID = '{repo_id}'")
        print("---------------------------------------------------")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

def upload_model_programmatic(hf_token: str, repo_id: str, local_folder: str = "outputs/fine_tuned_absa_model") -> bool:
    """Non-interactive upload function intended for automated pipelines."""
    try:
        import logging
        logger = logging.getLogger("HF_Uploader")
        logger.info(f"Uploading model to {repo_id}...")
        
        login(token=hf_token, add_to_git_credential=False)
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True)
        
        api.upload_folder(
            folder_path=local_folder,
            repo_id=repo_id,
            repo_type="model"
        )
        logger.info(f"✅ Successfully uploaded to {repo_id}")
        return True
    except Exception as e:
        import logging
        logger = logging.getLogger("HF_Uploader")
        logger.error(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    upload_model()
