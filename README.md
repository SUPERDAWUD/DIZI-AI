# DIZI AI Chatbot (Streamlit Edition)

This is a Streamlit-powered AI chatbot using open-source LLMs and intent-based responses.

## Features
- Chat with open-source LLMs (Falcon, GPT‑NeoX, StarCoder2, etc.)
- Intent-based instant replies (greetings, thanks, etc.)
- Model and personality switching (backend)
- Ready for deployment on Streamlit Cloud

### New (ChatGPT‑style) Enhancements
- Word‑by‑word streaming responses over SSE (`/chat/stream`)
- Stop streaming and Regenerate buttons in the web UI
- Lightweight Markdown rendering with code fences
- Local‑first generation (no OpenAI required). Falls back to LocalGPT when no APIs are configured
- “Super model” orchestration: gathers multiple evidence sources, applies local self‑consistency, and synthesizes a single answer

### Server UI (Flask)
Run the Flask server to use the full chat UI:
```
python server.py
```
Open http://localhost:5000 to chat.

Environment toggles:
- `AGGREGATION_MODE=parallel|priority` (default: `parallel`)
- `REASONING_MODE=off|auto|on` (optional)
- No OpenAI is used. If `GEMINI_API_KEY` or `HF_API_TOKEN` are set, they’ll be used; otherwise answers come from local models.

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
Made by D CORP inc

## Enhancements and Toggles

- ChatGPT-style streaming and UI: word-by-word SSE stream, Stop/Regenerate buttons, and lightweight Markdown rendering in the chat.
- Local-first text generation: text replies avoid OpenAI. If `GEMINI_API_KEY` or `HF_API_TOKEN` are set, they may be used; otherwise LocalGPT is used.
- Super model orchestration: combines multiple sources with local self-consistency to improve accuracy.
- Reasoning levels: set via Dev Panel or env `REASONING_LEVEL=low|medium|high`.
- Prefer local streaming: toggle in Dev Panel or set `PREFER_LOCAL_STREAM=true`.
- Optional Ollama backend: set `OLLAMA_MODEL` (default `llama3.1:8b-instruct`) and run Ollama at `http://localhost:11434`.
- Image generation: OpenAI/DALL·E removed; Replicate only.
