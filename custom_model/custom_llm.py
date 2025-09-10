import torch
import torch.nn as nn
import requests
import google.generativeai as genai
import os
import json
import time
from dotenv import load_dotenv
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

load_dotenv()
SERPAPI_KEY = os.getenv('SERPAPI_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
WOLFRAM_API_KEY = os.getenv('WOLFRAM_API_KEY')

class CustomLLM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Loader for model weights

def load_custom_llm(weights_path=None, device='auto'):
    """Load CustomLLM weights if available, otherwise fall back to random init."""
    auto_device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomLLM()
    if weights_path and os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=auto_device)
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[CustomLLM] Failed to load weights from {weights_path}: {e}. Using random weights.")
    else:
        print(f"[CustomLLM] No weights found at {weights_path}, using random weights.")
    model.to(auto_device)
    model.eval()
    return model

# --- Google Search Integration ---

def google_search(query, serpapi_key):
    params = {"q": query, "api_key": serpapi_key, "engine": "google"}
    try:
        if GoogleSearch is not None:
            results = GoogleSearch(params).get_dict()
        else:
            resp = requests.get("https://serpapi.com/search", params=params, timeout=5)
            resp.raise_for_status()
            results = resp.json()
        answer_box = results.get("answer_box", {})
        snippet = results.get("organic_results", [{}])[0].get("snippet", "")
        if "answer" in answer_box:
            return str(answer_box["answer"])
        elif snippet:
            return snippet
    except Exception:
        return None
    return None

# --- Gemini Integration ---

def gemini_generate(prompt, api_key, task='text'):
    genai.configure(api_key=api_key)
    model_name = 'gemini-pro' if task == 'text' else 'gemini-pro-vision'
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, 'text') else str(response)

# --- Text Generation ---

def generate_text(model, input_tensor, context=None):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    base_text = str(output.cpu().numpy())
    if context:
        try:
            gemini_key = context.get('gemini_key')
            if gemini_key:
                enhanced = gemini_generate(base_text, gemini_key, task='text')
                return enhanced
        except Exception:
            pass
    return base_text

# --- Code Generation ---

def generate_code(prompt, context=None):
    # Use Gemini or Hugging Face for code
    try:
        gemini_key = context.get('gemini_key') if context else None
        if gemini_key:
            return gemini_generate(prompt, gemini_key, task='text')
        # Add Hugging Face API fallback here
    except Exception:
        pass
    return "Code generation unavailable."

# --- Speech Generation ---

def generate_speech(prompt, context=None):
    # Use Gemini or other TTS APIs
    try:
        gemini_key = context.get('gemini_key') if context else None
        if gemini_key:
            return gemini_generate(prompt, gemini_key, task='text')
        # Add TTS API fallback here
    except Exception:
        pass
    return "Speech generation unavailable."

# --- Probability/Reasoning ---

def estimate_probability(prompt, context=None):
    # Use Gemini or custom logic
    try:
        gemini_key = context.get('gemini_key') if context else None
        if gemini_key:
            prob_text = gemini_generate(f"Estimate probability: {prompt}", gemini_key, task='text')
            return prob_text
    except Exception:
        pass
    return "Probability estimation unavailable."

# --- Utility: Everything Good ---

def everything_good(prompt, context=None):
    # Combine all features for a super response
    text = generate_text(context['model'], torch.randn(1, 768), context)
    code = generate_code(prompt, context)
    speech = generate_speech(prompt, context)
    prob = estimate_probability(prompt, context)
    google_result = google_search(prompt, context.get('serpapi_key', '')) if context and context.get('serpapi_key') else None
    return {
        'text': text,
        'code': code,
        'speech': speech,
        'probability': prob,
        'google': google_result
    }

# --- Context Memory ---
def get_user_context(username):
    chat_file = os.path.join('user_chats', f'{username}.json')
    if not os.path.exists(chat_file):
        return []
    with open(chat_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    # Return last 10 exchanges for context
    context = []
    for entry in history[-10:]:
        for msg in entry['chat']:
            context.append(f"{msg['user']}: {msg['message']}")
    return '\n'.join(context)

# --- Advanced Prompt Engineering ---
def build_prompt(user_msg, username=None, system_instructions=None, few_shot_examples=None):
    context = get_user_context(username) if username else ''
    prompt = ''
    if system_instructions:
        prompt += system_instructions + '\n'
    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Example: {ex}\n"
    if context:
        prompt += f"Context:\n{context}\n"
    prompt += f"User: {user_msg}\nAI:"
    return prompt

# --- Multi-turn Reasoning ---
def track_conversation_state(username, new_msg):
    # Append new message to user context
    chat_file = os.path.join('user_chats', f'{username}.json')
    if not os.path.exists(chat_file):
        history = []
    else:
        with open(chat_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    if history and 'chat' in history[-1]:
        history[-1]['chat'].append(new_msg)
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

# --- Knowledge Retrieval (RAG, Wikipedia, Web) ---
def retrieve_knowledge(query):
    # RAG: Use Google, Wikipedia, and local KB
    google_result = google_search(query, SERPAPI_KEY) if SERPAPI_KEY else None
    try:
        import wikipedia
        wiki_result = wikipedia.summary(query, sentences=2)
    except Exception:
        wiki_result = None
    # Add local KB lookup here if needed
    return {'google': google_result, 'wikipedia': wiki_result}

# --- Code Execution and Explanation ---
def execute_code(code, language='python'):
    if language == 'python':
        import io, contextlib
        f = io.StringIO()
        try:
            with contextlib.redirect_stdout(f):
                exec(code, {})
            return f.getvalue()
        except Exception as e:
            return f'Error: {e}'
    return 'Code execution only supported for Python.'

def explain_code(code, language='python'):
    # Use Gemini or LLM for explanation
    if GEMINI_API_KEY:
        return gemini_generate(f"Explain this {language} code: {code}", GEMINI_API_KEY, task='text')
    return 'Code explanation unavailable.'

# --- Math Solver (SymPy, WolframAlpha) ---
def solve_math(expr):
    try:
        import sympy
        result = sympy.sympify(expr)
        return str(result)
    except Exception:
        pass
    if WOLFRAM_API_KEY:
        # WolframAlpha API call
        url = f"http://api.wolframalpha.com/v1/result?appid={WOLFRAM_API_KEY}&i={expr}"
        try:
            resp = requests.get(url)
            if resp.ok:
                return resp.text
        except Exception:
            pass
    return 'Math solving unavailable.'

# --- Personality and Style Adaptation ---
def adapt_personality(response, personality):
    if personality == 'friendly':
        return f"😊 {response}"
    elif personality == 'formal':
        return f"Dear user, {response}"
    elif personality == 'sarcastic':
        return f"Oh, really? {response}"
    elif personality == 'creative':
        return f"✨ {response} ✨"
    elif personality == 'direct':
        return response
    return response

# --- Feedback Loop ---
def save_feedback(username, feedback):
    feedback_file = os.path.join('user_chats', f'{username}_feedback.json')
    if not os.path.exists(feedback_file):
        history = []
    else:
        with open(feedback_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    history.append({'feedback': feedback, 'timestamp': time.time()})
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
