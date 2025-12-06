import google.generativeai as genai
import os
import toml

# Load secrets
try:
    secrets = toml.load(".streamlit/secrets.toml")
    api_key = secrets.get("GOOGLE_API_KEY") or secrets.get("general", {}).get("GOOGLE_API_KEY")
    print(f"Key found: {api_key[:5]}...")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error loading secrets: {e}")
    exit()

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
