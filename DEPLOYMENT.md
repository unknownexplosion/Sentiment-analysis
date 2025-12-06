# 🚀 How to Deploy Securely to Streamlit Cloud

**Goal:** Publish your app to the internet without leaking your passwords.

---

## 🛑 Step 1: Safety Check
Make sure you are **NOT** committing your passwords to GitHub.
We already set this up, but verify your `.gitignore` file contains:
```
.streamlit/
.env
__pycache__/
```
*Result: Your local `secrets.toml` stays on your laptop. GitHub only sees the code.*

---

## ☁️ Step 2: Push to GitHub
1.  Create a new repository on GitHub.
2.  Push your code:
    ```bash
    git init
    git add .
    git commit -m "Final submission ready for deploy"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
    git push -u origin main
    ```

---

## 🔒 Step 3: Deploy & Set Secrets (The Important Part)
1.  Go to **[share.streamlit.io](https://share.streamlit.io/)**.
2.  Log in with GitHub.
3.  Click **"New App"** -> Select your repository.
4.  **BEFORE** clicking "Deploy", look for **"Advanced Settings"** (or the "Manage App" settings after deploy).
5.  Find the **"Secrets"** section.
6.  Paste your secrets there using the TOML format (just like your local file):

```toml
[general]
MONGO_URI = "mongodb+srv://noobdrawsdoodle_db_user:YOUR_PASSWORD@cluster0..."
GOOGLE_API_KEY = "AIzaSy..."
```
*(Copy these exact values from your local `.streamlit/secrets.toml` file)*

7.  Click **Save**.

---

## ✅ Result
*   Your App is live on the internet.
*   **The Inputs are Hidden:** Because we detected the secrets, the app will just show "✅ Google API Key Loaded" instead of the text box.
*   **Zero Leaks:** A user inspecting the webpage code will NEVER see your backend secrets. Only the Streamlit server knows them.
