import google.generativeai as genai
import os

# Load API Key from secrets if env var not set, or hardcode it for this check since I know it from previous turns
# Actually, I'll try to read it from the secrets file I created to be safe/consistent
try:
    with open(".streamlit/secrets.toml", "r") as f:
        for line in f:
            if "GOOGLE_API_KEY" in line:
                key = line.split('=')[1].strip().replace('"', '')
                genai.configure(api_key=key)
                break
except:
    print("Could not read secrets file.")

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
