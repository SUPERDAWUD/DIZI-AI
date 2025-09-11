import os
import json
import time
import torch
import torch.nn as nn
import requests
import google.generativeai as genai

try:
    from serpapi import GoogleSearch
except Exception:
    GoogleSearch = None


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


def _auto_device(device: str) -> str:
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def load_custom_llm(weights_path=None, device='auto'):
    device = _auto_device(device)
    model = CustomLLM()
    if weights_path and os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[CustomLLM] Failed to load weights from {weights_path}: {e}. Using random weights.")
    else:
        print(f"[CustomLLM] No weights found at {weights_path}, using random weights.")
    model.to(device)
    model.eval()
    return model


def google_search(query, serpapi_key: str):
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


def gemini_generate(prompt: str, api_key: str, task: str = 'text'):
    genai.configure(api_key=api_key)
    model_name = 'models/gemini-2.5-pro' if task == 'text' else 'models/gemini-2.5-pro'
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, 'text') else str(response)


def generate_text(model: CustomLLM, input_tensor: torch.Tensor, context=None):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    base_text = str(output.detach().cpu().numpy())
    if context:
        try:
            gemini_key = context.get('gemini_key')
            if gemini_key:
                return gemini_generate(base_text, gemini_key, task='text')
        except Exception:
            pass
    return base_text


def generate_code(prompt: str, context=None):
    try:
        gemini_key = context.get('gemini_key') if context else None
        if gemini_key:
            return gemini_generate(prompt, gemini_key, task='text')
    except Exception:
        pass
    return "Code generation unavailable."


def generate_speech(prompt: str, context=None):
    try:
        gemini_key = context.get('gemini_key') if context else None
        if gemini_key:
            return gemini_generate(prompt, gemini_key, task='text')
    except Exception:
        pass
    return "Speech generation unavailable."


def estimate_probability(prompt: str, context=None):
    try:
        gemini_key = context.get('gemini_key') if context else None
        if gemini_key:
            ans = gemini_generate(
                f"Estimate in one number [0-1] how likely this statement is true: {prompt}",
                gemini_key,
                task='text',
            )
            # Extract a number if possible
            import re
            m = re.search(r"0?\.\d+|1\.0+|1(?!\d)", ans)
            return float(m.group(0)) if m else ans
    except Exception:
        pass
    return None


def everything_good(prompt: str, context=None):
    text = None
    try:
        gemini_key = context.get('gemini_key') if context else None
        if gemini_key:
            text = gemini_generate(prompt, gemini_key, task='text')
    except Exception:
        text = None
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

def deliberate_reason(prompt: str, api_key: str = None, samples: int = 3):
    """
    Multi-sample deliberate reasoning using Gemini if available; falls back to simple echo.
    """
    tries = []
    try:
        if api_key:
            for style in ["concise", "direct", "detailed"][:samples]:
                p = (
                    f"Think step by step in a {style} way and produce the final answer at the end.\n"
                    f"Question: {prompt}\nFinal answer:"
                )
                tries.append(gemini_generate(p, api_key, task='text'))
        else:
            tries.append(prompt)
    except Exception:
        if not tries:
            tries.append(prompt)
    try:
        if api_key and len(tries) > 1:
            vote = gemini_generate(
                "Select the most accurate and helpful of these answers and output only that answer.\n\n" +
                "\n\n".join([f"Option {i+1}: {t}" for i, t in enumerate(tries)]) +
                "\n\nBest:",
                api_key,
                task='text'
            )
            return vote
    except Exception:
        pass
    return tries[0]

def super_generate(prompt: str, context=None):
    """
    Combines deliberate reasoning + retrieval (Google) + synthesis for a stronger response.
    """
    ctx = context or {}
    api_key = ctx.get('gemini_key') if isinstance(ctx, dict) else None
    base = everything_good(prompt, context)
    reasoned = deliberate_reason(prompt, api_key=api_key)
    try:
        if api_key:
            synthesis = gemini_generate(
                "Synthesize the final answer using the following pieces. Prefer factual items, be concise.\n\n" +
                f"Deliberate: {reasoned}\n\nText: {base.get('text')}\n\nCode: {base.get('code')}\n\n",
                api_key,
                task='text'
            )
            return synthesis
    except Exception:
        pass
    return reasoned or base.get('text')


def adapt_personality(response: str, personality: str):
    if personality == 'friendly':
        return f"Here to help: {response}"
    elif personality == 'formal':
        return f"Dear user, {response}"
    elif personality == 'sarcastic':
        return f"Oh, really? {response}"
    elif personality == 'creative':
        return f"✨ {response}"
    elif personality == 'direct':
        return response
    return response


def save_feedback(username: str, feedback: str):
    os.makedirs('user_chats', exist_ok=True)
    feedback_file = os.path.join('user_chats', f'{username}_feedback.json')
    if not os.path.exists(feedback_file):
        history = []
    else:
        with open(feedback_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    history.append({'feedback': feedback, 'timestamp': time.time()})
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


