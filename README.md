# DIZI AI Chatbot (Streamlit Edition)

This is a Streamlit-powered AI chatbot using open-source LLMs and intent-based responses.

## Features
- Chat with open-source LLMs (Falcon, GPT-NeoX, StarCoder2, etc.)
- Intent-based instant replies (greetings, thanks, etc.)
- Model and personality switching (backend)
- Ready for deployment on Streamlit Cloud

## How to Run Locally
1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

## How to Deploy on Streamlit Cloud
1. Push this project to a public GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in.
3. Click "New app", select your repo, and set `app.py` as the entry point.
4. Deploy and share your app!

## Notes
- Large models may require more memory than free Streamlit Cloud provides. For best results, use smaller models or run locally with a GPU.
- For advanced features (file upload, dev panel, etc.), use the Flask app or request a Streamlit UI upgrade.

---
Made with ❤️ by DIZI CORP
