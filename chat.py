import os
import json
import torch
import random
import difflib
import requests
import sympy
import re
import threading
from sympy import symbols, Eq, solve, simplify, expand, factor, diff, integrate
from dotenv import load_dotenv
from serpapi import GoogleSearch
from chatbot_model import NeuralNet
from utils import bag_of_words, tokenize, start_generator_api
import pytz
from datetime import datetime

# Load keys
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Load intents
with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# Load knowledge base
with open("data/knowledge_base.json", "r", encoding="utf-8") as f:
    kb_data = json.load(f)
    knowledge_base = {item["question"].lower(): item["answer"] for item in kb_data.get("questions", [])}

# Load model
FILE = "model.pth"
if torch.cuda.is_available():
    data = torch.load(FILE)
else:
    data = torch.load(FILE, map_location=torch.device("cpu"))
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

USER_DATA_FILE = "user_data.json"
USER_QUESTIONS_FILE = "user_questions.json"
USER_LOCATION_FILE = "user_location.json"
user_data_lock = threading.Lock()
user_questions_lock = threading.Lock()
user_location_lock = threading.Lock()

def update_user_data(username):
    username = username.strip().capitalize()
    with user_data_lock:
        try:
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception:
            users = []
        if username not in users:
            users.append(username)
            with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(users, f, indent=2)
            return False  # New user
        return True  # Existing user

def update_user_questions(username, question):
    username = username.strip().capitalize()
    with user_questions_lock:
        try:
            with open(USER_QUESTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        if username not in data:
            data[username] = []
        if question not in data[username]:
            data[username].append(question)
            with open(USER_QUESTIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

def get_user_location(ip_address):
    """Get location info from IP using ip-api.com (free, no key needed)."""
    try:
        resp = requests.get(f"http://ip-api.com/json/{ip_address}")
        if resp.ok:
            data = resp.json()
            if data.get("status") == "success":
                return {
                    "city": data.get("city"),
                    "region": data.get("regionName"),
                    "country": data.get("country"),
                    "lat": data.get("lat"),
                    "lon": data.get("lon")
                }
    except Exception:
        pass
    return None

def update_user_location(username, ip_address):
    username = username.strip().capitalize()
    with user_location_lock:
        try:
            with open(USER_LOCATION_FILE, "r", encoding="utf-8") as f:
                locations = json.load(f)
        except Exception:
            locations = {}
        if username not in locations:
            loc = get_user_location(ip_address)
            if loc:
                locations[username] = loc
                with open(USER_LOCATION_FILE, "w", encoding="utf-8") as f:
                    json.dump(locations, f, indent=2)
                return loc
        return locations.get(username)

def get_user_location_from_file(username):
    username = username.strip().capitalize()
    try:
        with open(USER_LOCATION_FILE, "r", encoding="utf-8") as f:
            locations = json.load(f)
        return locations.get(username)
    except Exception:
        return None

def get_local_time(location):
    # Use city/country to get timezone and local time
    try:
        import requests
        city = location.get("city")
        country = location.get("country")
        if city and country:
            # Use worldtimeapi.org for timezone lookup
            resp = requests.get(f"http://worldtimeapi.org/api/timezone")
            if resp.ok:
                timezones = resp.json()
                # Try to find a matching timezone
                for tz in timezones:
                    if city.replace(" ", "_") in tz or country.replace(" ", "_") in tz:
                        time_resp = requests.get(f"http://worldtimeapi.org/api/timezone/{tz}")
                        if time_resp.ok:
                            dt = time_resp.json().get("datetime")
                            if dt:
                                return f"The local time in {city}, {country} is {dt[:19].replace('T', ' ')}."
    except Exception:
        pass
    return "Sorry, I couldn't determine your local time."

def get_weather(location):
    # Use SerpAPI to get weather from Google
    api_key = os.getenv("SERPAPI_KEY")
    city = location.get("city")
    country = location.get("country")
    if not api_key or not city:
        return "Weather service is not configured."
    try:
        from serpapi import GoogleSearch
        query = f"weather in {city}, {country}"
        search = GoogleSearch({
            "q": query,
            "api_key": api_key,
            "engine": "google"
        })
        results = search.get_dict()
        weather_box = results.get("answer_box", {})
        if "temperature" in weather_box and "unit" in weather_box:
            temp = weather_box["temperature"]
            unit = weather_box["unit"]
            desc = weather_box.get("weather", "")
            return f"The weather in {city}, {country} is {desc}, {temp}{unit}."
        elif "snippet" in weather_box:
            return f"Weather in {city}, {country}: {weather_box['snippet']}"
        else:
            return "Sorry, I couldn't get the weather for your location."
    except Exception:
        return "Sorry, I couldn't get the weather for your location."

# Intent response engine
def get_intent_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.6:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                # Add personality: sometimes add a prefix or emoji
                import random
                response = random.choice(intent["responses"])
                fun_prefixes = [
                    "🤖 ", "✨ ", "[AI says] ", "", "", "", "", "", "", ""  # mostly empty for subtlety
                ]
                return random.choice(fun_prefixes) + response
    return None

# Knowledge base fuzzy match
def get_from_knowledge_base(msg, threshold=0.85):
    msg = msg.strip().lower()
    if msg in knowledge_base:
        return knowledge_base[msg]
    match = difflib.get_close_matches(msg, list(knowledge_base.keys()), n=1, cutoff=threshold)
    if match:
        return knowledge_base[match[0]]
    return None

# Math handler (basic & symbolic)
def handle_math(msg):
    x, y = symbols("x y")
    msg = msg.lower().strip().replace("^", "**")

    try:
        if re.fullmatch(r"[0-9\.\+\-\*/\(\)\s\^]+", msg):
            result = eval(msg, {"__builtins__": None}, {})
            return f"The answer is {result}"
        if msg.startswith("solve"):
            expr = msg.replace("solve", "").strip()
            if "=" in expr:
                lhs, rhs = expr.split("=")
                sol = solve(Eq(sympy.sympify(lhs), sympy.sympify(rhs)))
            else:
                sol = solve(sympy.sympify(expr))
            return f"Solution: {sol}"
        if msg.startswith("simplify"):
            return f"Simplified: {simplify(msg.replace('simplify', '').strip())}"
        if msg.startswith("expand"):
            return f"Expanded: {expand(msg.replace('expand', '').strip())}"
        if msg.startswith("factor"):
            return f"Factored: {factor(msg.replace('factor', '').strip())}"
        if msg.startswith("differentiate"):
            return f"Derivative: {diff(msg.replace('differentiate', '').strip())}"
        if msg.startswith("integrate"):
            return f"Integral: {integrate(msg.replace('integrate', '').strip())}"
    except Exception as e:
        return f"Math error: {e}"
    return None

# Image generation (Replicate API)
def generate_image_from_prompt(prompt):
    import replicate
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        return "Replicate API token is missing."
    try:
        # Use a popular, free, and public model: stability-ai/sdxl
        output = replicate.run(
            "stability-ai/sdxl:9fa24d7e8cf3e4c0b7e7b2e0b1e5e1b6b6e1b6e1b6e1b6e1b6e1b6e1b6e1b6e1b6",
            input={"prompt": prompt},
            api_token=replicate_token
        )
        if output and isinstance(output, list):
            return f"Here is your image: {output[0]}"
        elif isinstance(output, str):
            return f"Here is your image: {output}"
        return "Image generation failed."
    except Exception as e:
        return f"Image error: {e}"





# Google fallback
def ask_google(query):
    if not SERPAPI_KEY:
        return "Google API key is missing."
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google"
        })
        results = search.get_dict()
        answer_box = results.get("answer_box", {})
        snippet = results.get("organic_results", [{}])[0].get("snippet", "")

        if "answer" in answer_box:
            return f"Here’s what I found: {str(answer_box['answer'])}"
        elif snippet:
            return f"Here’s a quick answer: {snippet}"
        else:
            return "I searched high and low, but nothing turned up. Want to try rephrasing?"
    except Exception as e:
        return f"Search error: {e}"

# Main response router
def get_response(msg, username=None, ip_address=None):
    msg_lower = msg.strip().lower()

    # User tracking
    if username:
        update_user_data(username)
        update_user_questions(username, msg)
        if ip_address:
            update_user_location(username, ip_address)

    # 1. Math (always highest priority)
    math = handle_math(msg_lower)
    if math:
        return math
    # 2. Image prompt (always high priority)
    if "generate image of" in msg_lower or "create image of" in msg_lower:
        parts = re.split(r"generate image of|create image of", msg_lower, maxsplit=1)
        if len(parts) > 1:
            prompt = parts[1].strip()
            return generate_image_from_prompt(prompt)
        else:
            return "What image would you like me to create?"
    # 3. Generator API fallback (for code/speech generation)
    if any(word in msg_lower for word in ["generate code", "write code", "code for", "generate speech", "write speech", "speech for", "make a speech", "make speech"]):
        try:
            if "code" in msg_lower:
                resp = requests.post("http://localhost:5050/generate/code", json={"prompt": msg})
                if resp.ok:
                    code = resp.json().get("code", "")
                    if code and not code.startswith("# Sorry"):
                        return f"Here's some code for you:\n{code}"
                    else:
                        return "I couldn't find a good code template for that topic yet, but I'm learning!"
            elif "speech" in msg_lower:
                topic = msg
                for prefix in ["make a speech about", "write a speech about", "generate speech about", "speech for", "make speech about", "speech about", "write speech about"]:
                    if msg_lower.startswith(prefix):
                        topic = msg_lower.replace(prefix, "").strip()
                        break
                resp = requests.post("http://localhost:5050/generate/speech", json={"topic": topic, "audience": "everyone"})
                if resp.ok:
                    speech = resp.json().get("speech", "")
                    if speech:
                        followup_resp = requests.post("http://localhost:5050/generate/followup", json={})
                        if followup_resp.ok:
                            followup = followup_resp.json().get("followup", "")
                            return f"Here's a speech draft:\n{speech}\n\n{followup}"
                        return f"Here's a speech draft:\n{speech}"
                    else:
                        return "I couldn't generate a speech for that topic yet, but I'm working on it!"
        except Exception as e:
            return f"Generator API error: {e}"
    # 4. User's name (before intent)
    if username and any(q in msg_lower for q in ["what is my name", "what's my name", "do you know my name", "who am i"]):
        return f"Your name is {username}."
    # 5. User's location (before intent)
    if username and any(q in msg_lower for q in ["where do i live", "what's my location", "what is my location", "where am i"]):
        loc = get_user_location_from_file(username)
        if loc:
            city = loc.get("city")
            country = loc.get("country")
            return f"You are in {city}, {country}."
        else:
            return "I don't have your location yet."
    # 6. Local time (before intent)
    if username and any(q in msg_lower for q in ["what time is it", "what's the time", "current time", "local time"]):
        loc = get_user_location_from_file(username)
        if loc:
            return get_local_time(loc)
        else:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return f"I don't have your location, but my current time is {now}."
    # 7. Weather (before intent)
    if username and ("weather" in msg_lower or "what's the weather" in msg_lower or "what is the weather" in msg_lower):
        loc = get_user_location_from_file(username)
        if loc:
            return get_weather(loc)
        else:
            return "I don't have your location yet."
    # 8. Intent classification (after all special cases)
    intent = get_intent_response(msg)
    if intent:
        return intent
    # 9. Knowledge base
    kb = get_from_knowledge_base(msg)
    if kb:
        return kb
    # 10. Google fallback
    return ask_google(msg)

# Optional CLI mode
if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        start_generator_api()
        print("DIZI AI (CLI Mode). Type 'quit' to exit.")
        username = input("Enter your name: ").strip().capitalize()
        update_user_data(username)
        while True:
            msg = input(f"{username}: ")
            if msg.lower() == "quit":
                break
            update_user_questions(username, msg)
            reply = get_response(msg, username=username)
            print("DIZI:", reply)



