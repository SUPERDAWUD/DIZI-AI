from flask import Flask, request, jsonify, session
import os

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dizi_secret')

# Import transformers only inside the route to avoid crashing the whole app if there's a problem

def get_code_gen():
    try:
        import torch
        from transformers import pipeline, set_seed
        set_seed(42)
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("text-generation", model="Salesforce/codegen-350M-mono", device=device)
    except Exception as e:
        print(f"CodeGen model load error: {e}")
        return None

def get_text_gen():
    try:
        import torch
        from transformers import pipeline, set_seed
        set_seed(42)
        device = 0 if torch.cuda.is_available() else -1
        # Try a more capable instruction-following model if available
        try:
            return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device=device)
        except Exception:
            try:
                return pipeline("text2text-generation", model="google/flan-t5-base", device=device)
            except Exception:
                return pipeline("text-generation", model="gpt2", device=device)
    except Exception as e:
        print(f"TextGen model load error: {e}")
        return None

# Helper to store last topic
FOLLOWUP_KEY = 'last_topic'

@app.route("/generate/code", methods=["POST"])
def generate_code():
    data = request.get_json()
    topic = data.get("prompt", "")
    code_gen = get_code_gen()
    if code_gen:
        prompt = f"# Python function to {topic}\n"
        result = code_gen(prompt, max_length=128, num_return_sequences=1, temperature=0.3)
        code = result[0]["generated_text"].replace(prompt, "").strip()
        return jsonify({"code": code})
    # Fallback
    if "bubble sort" in topic.lower():
        code = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr"""
    else:
        code = f"# Sorry, I don't have a template for '{topic}' yet."
    return jsonify({"code": code})

@app.route("/generate/speech", methods=["POST"])
def generate_speech():
    data = request.get_json()
    topic = data.get("topic", "something important")
    audience = data.get("audience", "everyone")
    tone = data.get("tone", "informative")
    text_gen = get_text_gen()
    # Store topic in session for follow-up
    session[FOLLOWUP_KEY] = topic
    if text_gen:
        # Use a more explicit prompt for instruction models
        prompt = f"Write a short, {tone} speech about {topic} for {audience}. The speech should be clear, on-topic, and inspiring.\n"
        # Use text2text-generation if using flan-t5-base, else text-generation
        if text_gen.task == "text2text-generation":
            result = text_gen(prompt, max_length=120, num_return_sequences=1)
            speech = result[0]["generated_text"] if "generated_text" in result[0] else result[0]["generated_text"]
        else:
            result = text_gen(prompt, max_length=120, num_return_sequences=1, temperature=0.7)
            speech = result[0]["generated_text"].replace(prompt, "").strip()
        return jsonify({"speech": speech})
    # Fallback
    speech = (
        f"Dear {audience},\n\n"
        f"Today I want to speak about {topic}. This topic touches the heart of our community... "
        f"Together, we can make a difference.\n\nThank you."
    )
    return jsonify({"speech": speech})

@app.route("/generate/followup", methods=["POST"])
def generate_followup():
    # Use last topic from session
    topic = session.get(FOLLOWUP_KEY, "the previous topic")
    text_gen = get_text_gen()
    if text_gen:
        prompt = f"Write a follow-up question or comment for a speech about {topic}:\n"
        result = text_gen(prompt, max_length=60, num_return_sequences=1, temperature=0.8)
        followup = result[0]["generated_text"].replace(prompt, "").strip()
        return jsonify({"followup": followup})
    # Fallback
    return jsonify({"followup": f"Would you like to hear more about {topic}?"})

if __name__ == "__main__":
    app.run(port=5050)
