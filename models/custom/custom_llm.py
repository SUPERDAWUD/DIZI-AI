import json
import os
import re
import time

import torch
import torch.nn as nn


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
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_custom_llm(weights_path=None, device="auto"):
    device = _auto_device(device)
    model = CustomLLM()
    if weights_path and os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state, strict=False)
        except Exception as exc:
            print(f"[CustomLLM] Failed to load weights from {weights_path}: {exc}. Using random weights.")
    else:
        print(f"[CustomLLM] No weights found at {weights_path}, using random weights.")
    model.to(device)
    model.eval()
    return model


def generate_text(model: CustomLLM, input_tensor: torch.Tensor, context=None):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    values = output.detach().cpu().numpy().flatten().tolist()[:16]
    return " ".join(str(value) for value in values)


def generate_code(prompt: str, context=None):
    prompt = str(prompt or "").strip()
    if not prompt:
        return "Provide a local code request to generate or explain code."
    return (
        "```python\n"
        "def solution():\n"
        "    raise NotImplementedError('Load a local coding GGUF model for full code generation.')\n"
        "```\n\n"
        f"Local-only note: no remote code model was called. Request: {prompt}"
    )


def generate_speech(prompt: str, context=None):
    prompt = str(prompt or "").strip()
    return f"Local speech text: {prompt}" if prompt else "Provide text for local speech generation."


def estimate_probability(prompt: str, context=None):
    text = str(prompt or "").lower()
    if not text:
        return None
    strong_markers = {"always", "never", "guaranteed", "impossible"}
    cautious_markers = {"may", "might", "could", "likely", "probably"}
    words = set(re.findall(r"[a-z']+", text))
    if words & strong_markers:
        return 0.45
    if words & cautious_markers:
        return 0.65
    return 0.5


def everything_good(prompt: str, context=None):
    return {
        "text": deliberate_reason(prompt),
        "code": generate_code(prompt, context),
        "speech": generate_speech(prompt, context),
        "probability": estimate_probability(prompt, context),
        "retrieval": None,
    }


def _compact_prompt(prompt: str, max_words: int = 80) -> str:
    words = str(prompt or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def deliberate_reason(prompt: str, samples: int = 3):
    """Local-only lightweight reasoning fallback."""
    prompt = str(prompt or "").strip()
    if not prompt:
        return "No prompt provided."

    if "?" in prompt:
        return f"Local answer: {_compact_prompt(prompt)}"
    return f"Local response: {_compact_prompt(prompt)}"


def super_generate(prompt: str, context=None):
    """Generate through local helpers only."""
    return deliberate_reason(prompt)


def adapt_personality(response: str, personality: str):
    if personality == "friendly":
        return f"Here to help: {response}"
    if personality == "formal":
        return f"Dear user, {response}"
    if personality == "sarcastic":
        return f"Oh, really? {response}"
    if personality == "creative":
        return f"Creative response: {response}"
    if personality == "direct":
        return response
    return response


def save_feedback(username: str, feedback: str):
    os.makedirs("user_chats", exist_ok=True)
    feedback_file = os.path.join("user_chats", f"{username}_feedback.json")
    if not os.path.exists(feedback_file):
        history = []
    else:
        with open(feedback_file, "r", encoding="utf-8") as handle:
            history = json.load(handle)
    history.append({"feedback": feedback, "timestamp": time.time()})
    with open(feedback_file, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
