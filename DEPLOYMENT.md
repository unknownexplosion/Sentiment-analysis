# Deployment Guide for Streamlit Community Cloud

Since your code is already on GitHub, deploying your dashboard is very easy and free!

## Prerequisites
1. Ensure your GitHub repository is **public** (or you have a Streamlit Community Cloud account linked to it).
2. Your repository should contain:
   - `app.py` (The main dashboard file)
   - `requirements.txt` (List of dependencies)
   - `outputs/*.csv` (The data files to display)

(We have already set all of this up!)

## Steps to Deploy

1. **Go to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io/).
   - Click **"Sign up"** (or "Log in") and use your **GitHub account**.

2. **Create a New App:**
   - Click the **"New app"** button (top right).
   - If prompted, authorize Streamlit to access your GitHub repositories.

3. **Configure the App:**
   - **Repository:** Select your repository: `unknownexplosion/Sentiment-analysis` (or search for it).
   - **Branch:** `main`
   - **Main file path:** `app.py`

4. **Deploy:**
   - Click **"Deploy!"**.
   - Watch the logs as it builds your environment (it downloads the libraries from `requirements.txt`).
   - In a few minutes, your app will be live! 🚀

## Notes
- **Data Updates:** The app currently reads from the `outputs/` CSV files. On Streamlit Cloud, the file system is ephemeral. If you run the pipeline *there* (which we haven't set up a button for yet), the changes won't be saved back to GitHub automatically.
- **For now:** Run the pipeline locally on your machine, commit the new CSVs to GitHub, and the Streamlit app will update automatically!
