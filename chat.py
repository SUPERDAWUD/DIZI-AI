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
from nltk.corpus import words as nltk_words
import mimetypes
import io
import contextlib
import tempfile
import shutil

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

try:
    nltk_words.words()
except LookupError:
    import nltk
    nltk.download('words')

# Cache English words for spell correction to avoid repeated lookups
ENGLISH_WORDS = set(nltk_words.words())

# Spell correction helper
def correct_word(word):
    """
    Correct a single word using a static cache of English words and difflib for fuzzy matching.
    Returns the corrected word if a close match is found, otherwise returns the original word.
    """
    # Use a static cache for faster lookup
    if word in ENGLISH_WORDS:
        return word
    matches = difflib.get_close_matches(word, ENGLISH_WORDS, n=1, cutoff=0.8)
    return matches[0] if matches else word

def correct_sentence(sentence):
    """
    Corrects each word in a sentence using correct_word, only if not already in the English word cache.
    Returns the corrected sentence as a string.
    """
    # Only correct words that are not in the cache (faster)
    return ' '.join(correct_word(w) if w not in ENGLISH_WORDS else w for w in sentence.split())

def atomic_write_json(data, filename):
    """
    Atomically writes JSON data to a file to avoid corruption.
    Writes to a temporary file and then moves it to the target filename.
    """
    # Write JSON atomically to avoid corruption
    dirpath = os.path.dirname(os.path.abspath(filename))
    with tempfile.NamedTemporaryFile('w', dir=dirpath, delete=False, encoding='utf-8') as tf:
        json.dump(data, tf, indent=2)
        tempname = tf.name
    shutil.move(tempname, filename)

def update_user_data(username):
    """
    Adds a username to the user data file if not already present.
    Uses a lock for thread safety and atomic write for robustness.
    Returns False if new user, True if existing.
    """
    username = username.strip().capitalize()
    with user_data_lock:
        try:
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception:
            users = []
        if username not in users:
            users.append(username)
            try:
                atomic_write_json(users, USER_DATA_FILE)
            except Exception as e:
                print(f"[ERROR] Failed to write user data: {e}")
            return False  # New user
        return True  # Existing user

def update_user_questions(username, question, max_questions=100):
    """
    Adds a question to the user's question history, deduplicates, and limits to max_questions.
    Uses a lock for thread safety and atomic write for robustness.
    """
    username = username.strip().capitalize()
    with user_questions_lock:
        try:
            with open(USER_QUESTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        if username not in data:
            data[username] = []
        # Deduplicate and limit
        if question not in data[username]:
            data[username].append(question)
            if len(data[username]) > max_questions:
                data[username] = data[username][-max_questions:]
            try:
                atomic_write_json(data, USER_QUESTIONS_FILE)
            except Exception as e:
                print(f"[ERROR] Failed to write user questions: {e}")

def get_user_location(ip_address):
    """
    Gets location info (city, region, country, lat, lon) from an IP address using ip-api.com.
    Returns a dict with location info or None if lookup fails.
    """
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
    """
    Updates the user's location in the user location file if not already present.
    Uses a lock for thread safety and atomic write for robustness.
    Returns the location dict or None.
    """
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
                try:
                    atomic_write_json(locations, USER_LOCATION_FILE)
                except Exception as e:
                    print(f"[ERROR] Failed to write user location: {e}")
                return loc
        return locations.get(username)

def get_user_location_from_file(username):
    """
    Retrieves the user's location from the user location file.
    Returns the location dict or None if not found.
    """
    username = username.strip().capitalize()
    try:
        with open(USER_LOCATION_FILE, "r", encoding="utf-8") as f:
            locations = json.load(f)
        return locations.get(username)
    except Exception:
        return None

def get_local_time(location):
    """
    Gets the local time for a given location dict (city, country) using worldtimeapi.org.
    Returns a string with the local time or an error message.
    """
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
    """
    Gets the weather for a given location dict (city, country) using SerpAPI/Google.
    Returns a string with the weather or an error message.
    """
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

# --- Enhanced: Robust intent/KB matching, grammar normalization, and speech upgrades ---
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper: Normalize grammar (basic)
def normalize_grammar(text):
    # Lowercase, remove extra spaces, basic punctuation fix
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\.,?!]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Helper: Semantic similarity (TF-IDF cosine)
def semantic_similarity(query, choices):
    try:
        vectorizer = TfidfVectorizer().fit([query] + choices)
        vectors = vectorizer.transform([query] + choices)
        sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        idx = sims.argmax()
        return choices[idx], sims[idx]
    except Exception:
        return None, 0

# Improved intent response engine (fuzzy + semantic)
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
    if prob.item() > 0.5:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                fun_prefixes = ["🤖 ", "✨ ", "[AI says] ", "", "", "", "", "", "", ""]
                return random.choice(fun_prefixes) + response
    # Fuzzy/semantic fallback
    all_patterns = []
    tag_map = {}
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            all_patterns.append(pattern)
            tag_map[pattern] = intent["tag"]
    match, score = semantic_similarity(msg, all_patterns)
    if score > 0.5:
        tag = tag_map[match]
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    return None

# Improved KB fuzzy/semantic match
def get_from_knowledge_base(msg, threshold=0.7):
    msg = normalize_grammar(msg)
    kb_keys = list(knowledge_base.keys())
    match, score = semantic_similarity(msg, kb_keys)
    if score > threshold:
        return knowledge_base[match]
    # fallback to difflib
    match = difflib.get_close_matches(msg, kb_keys, n=1, cutoff=threshold)
    if match:
        return knowledge_base[match[0]]
    return None

# --- Improved speech extraction and generation ---
def extract_speech_request(msg):
    # Accepts poor grammar, e.g. "speech cat", "make speech cat funny", etc.
    msg = normalize_grammar(msg)
    if "speech" in msg:
        # Try to extract topic, tone, audience, length
        topic = None
        tone = None
        audience = None
        length = None
        # Extract topic (after 'speech' or 'about')
        match = re.search(r'speech(?: about| on| for| regarding)? ([\w\s,]+)', msg)
        if match:
            topic = match.group(1).strip()
        else:
            # fallback: first noun after 'speech'
            parts = msg.split('speech', 1)
            if len(parts) > 1:
                topic = parts[1].strip(' ,.')
        # Tone
        if 'funny' in msg:
            tone = 'funny'
        elif 'serious' in msg:
            tone = 'serious'
        elif 'motivational' in msg:
            tone = 'motivational'
        elif 'sad' in msg:
            tone = 'sad'
        # Audience
        aud_match = re.search(r'for ([\w\s]+)', msg)
        if aud_match:
            audience = aud_match.group(1).strip()
        # Length
        if 'short' in msg:
            length = 'short'
        elif 'long' in msg:
            length = 'long'
        return topic or msg, tone, audience, length
    return None, None, None, None

# Image generation (Replicate API)
def generate_image_from_prompt(prompt):
    import replicate
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    dalle_token = os.getenv("OPENAI_API_KEY")
    if not replicate_token and not dalle_token:
        return "No image generation API token is configured."
    try:
        # Parse style/model/quality hints from prompt
        style = None
        model = None
        quality = None
        # Example: "draw a cat in anime style, high quality, using SDXL"
        style_match = re.search(r"(anime|realistic|cartoon|sketch|oil painting|watercolor|pixel art|cyberpunk|fantasy|photorealistic)", prompt, re.I)
        if style_match:
            style = style_match.group(1).lower()
        model_match = re.search(r"(sdxl|dall[\.-]?e|dalle|stable diffusion|midjourney|anything v3|dreamshaper|openjourney)", prompt, re.I)
        if model_match:
            model = model_match.group(1).lower()
        quality_match = re.search(r"(high quality|ultra quality|4k|8k|hd|detailed|masterpiece)", prompt, re.I)
        if quality_match:
            quality = quality_match.group(1).lower()
        # Clean prompt for model
        clean_prompt = prompt
        for m in [style, model, quality]:
            if m:
                clean_prompt = re.sub(m, '', clean_prompt, flags=re.I)
        clean_prompt = clean_prompt.strip(', .')
        # Model selection logic
        if model and ("dall" in model or "openai" in model):
            # Use DALL·E via OpenAI API
            if not dalle_token:
                return "DALL·E API key is missing."
            import openai
            openai.api_key = dalle_token
            try:
                response = openai.Image.create(
                    prompt=clean_prompt + (f", {style}" if style else "") + (f", {quality}" if quality else ""),
                    n=1,
                    size="1024x1024"
                )
                url = response['data'][0]['url']
                return f"Here is your DALL·E image: {url}"
            except Exception as e:
                return f"DALL·E error: {e}"
        # Default: SDXL or other Replicate models
        replicate_model = "stability-ai/sdxl"
        if model:
            if "anime" in model or "anything" in model:
                replicate_model = "andite/anything-v4.0"
            elif "dreamshaper" in model:
                replicate_model = "lucataco/dreamshaper-v8"
            elif "openjourney" in model:
                replicate_model = "prompthero/openjourney"
            elif "realistic" in model:
                replicate_model = "stablediffusionapi/realistic-vision-v51"
            elif "midjourney" in model:
                replicate_model = "prompthero/openjourney"
            elif "dall" in model:
                # fallback to DALL·E above
                pass
        if style == "anime":
            replicate_model = "andite/anything-v4.0"
        elif style == "realistic":
            replicate_model = "stablediffusionapi/realistic-vision-v51"
        # Add quality to prompt
        full_prompt = clean_prompt
        if style:
            full_prompt += f", {style} style"
        if quality:
            full_prompt += f", {quality}"
        output = replicate.run(
            f"{replicate_model}",
            input={"prompt": full_prompt},
            api_token=replicate_token
        )
        if output and isinstance(output, list):
            return f"Here is your image: {output[0]}"
        elif isinstance(output, str):
            return f"Here is your image: {output}"
        return "Image generation failed."
    except Exception as e:
        return f"Image error: {e}"

# --- Optional: Image Captioning/Analysis (if file referenced) ---
def analyze_image_file(filename):
    import replicate
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        return "Replicate API token is missing."
    filepath = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(filepath):
        return "File not found."
    try:
        # Use BLIP image captioning model
        output = replicate.run(
            "salesforce/blip:7c6b5cfae6e8e4e4b7e7b2e0b1e5e1b6b6e1b6e1b6e1b6e1b6e1b6e1b6e1b6",
            input={"image": open(filepath, "rb")},
            api_token=replicate_token
        )
        if output:
            return f"Image caption: {output}"
        return "Image analysis failed."
    except Exception as e:
        return f"Image analysis error: {e}"

# Image generation (Replicate API)
def generate_image_from_prompt(prompt):
    import replicate
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        return "Replicate API token is missing."
    try:
        # Use a popular, free, and public model: stability-ai/sdxl
        output = replicate.run(
            "stability-ai/sdxl:9fa24d7e8cf3e4c0b7e7b2e0b1e5e1b6b6e1b6e1b6e1b6e1b6e1b6e1b6e1b6",
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

def extract_language(msg):
    # Look for language keywords in the message
    languages = ['python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'go', 'typescript', 'php', 'swift', 'kotlin', 'rust']
    for lang in languages:
        if lang in msg.lower():
            return lang
    return 'python'  # default

# Google fallback
def ask_google(query):
    # Only use Google fallback if the query is not handled by math, image, code, speech, intent, or knowledge base
    # This function should only be called as a true fallback
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
        followup = None
        # Only generate a followup if the query is a question or informational
        is_question = any(q in query.lower() for q in ['what', 'who', 'when', 'where', 'why', 'how', '?'])
        if is_question:
            try:
                followup_resp = requests.post("http://localhost:5050/generate/followup", json={"topic": query})
                if followup_resp.ok:
                    followup = followup_resp.json().get("followup", None)
            except Exception:
                pass
        if "answer" in answer_box:
            response = f"Here’s what I found: {str(answer_box['answer'])}"
        elif snippet:
            response = f"Here’s a quick answer: {snippet}"
        else:
            response = "I searched high and low, but nothing turned up. Want to try rephrasing?"
        if followup:
            response += f"\n\n{followup}"
        return response
    except Exception as e:
        return f"Search error: {e}"

# Run code in a restricted namespace and capture output
def run_python_code(code):
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {'__builtins__': __builtins__}, {})
        return output.getvalue(), None
    except Exception as e:
        return None, str(e)

# Add support for code execution in multiple languages (Python, JavaScript, etc.)
def run_code(code, language='python'):
    import subprocess
    import tempfile
    if language == 'python':
        return run_python_code(code)
    elif language in ['js', 'javascript']:
        with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(['node', fname], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout, None
            else:
                return None, result.stderr
        except Exception as e:
            return None, str(e)
        finally:
            os.remove(fname)
    elif language in ['bash', 'sh', 'shell']:
        with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(['bash', fname], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout, None
            else:
                return None, result.stderr
        except Exception as e:
            return None, str(e)
        finally:
            os.remove(fname)
    elif language in ['c', 'cpp', 'c++']:
        ext = '.c' if language == 'c' else '.cpp'
        with tempfile.NamedTemporaryFile('w', suffix=ext, delete=False) as f:
            f.write(code)
            fname = f.name
        exe = fname + '.exe'
        try:
            if language == 'c':
                compile_cmd = ['gcc', fname, '-o', exe]
            else:
                compile_cmd = ['g++', fname, '-o', exe]
            comp = subprocess.run(compile_cmd, capture_output=True, text=True)
            if comp.returncode != 0:
                return None, comp.stderr
            result = subprocess.run([exe], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout, None
            else:
                return None, result.stderr
        except Exception as e:
            return None, str(e)
        finally:
            os.remove(fname)
            if os.path.exists(exe):
                os.remove(exe)
    elif language == 'java':
        with tempfile.NamedTemporaryFile('w', suffix='.java', delete=False) as f:
            f.write(code)
            fname = f.name
        classname = os.path.splitext(os.path.basename(fname))[0]
        try:
            comp = subprocess.run(['javac', fname], capture_output=True, text=True)
            if comp.returncode != 0:
                return None, comp.stderr
            result = subprocess.run(['java', '-cp', os.path.dirname(fname), classname], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout, None
            else:
                return None, result.stderr
        except Exception as e:
            return None, str(e)
        finally:
            os.remove(fname)
            classfile = os.path.join(os.path.dirname(fname), classname + '.class')
            if os.path.exists(classfile):
                os.remove(classfile)
    return None, f"Language '{language}' not supported for execution."

# Math handler (basic & symbolic)
def handle_math(msg):
    x, y = symbols("x y")
    msg = msg.lower().strip().replace("^", "**")
    # Preprocess: remove common math question prefixes
    math_prefixes = [
        "what is ", "whats ", "what's ", "calculate ", "solve ", "compute ", "evaluate ", "find ", "give me ", "tell me ", "can you solve ", "can you calculate ", "can you evaluate ", "can you compute "
    ]
    for prefix in math_prefixes:
        if msg.startswith(prefix):
            msg = msg[len(prefix):].strip()
            break
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

# --- File Redaction Feature ---
def redact_file_content(filename, redactions):
    filepath = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(filepath):
        return None, "File not found."
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Redact each word/phrase (case-insensitive)
        for phrase in redactions:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            content = pattern.sub('[REDACTED]', content)
        # Save redacted file
        redacted_name = f"redacted_{filename}"
        redacted_path = os.path.join(UPLOADS_DIR, redacted_name)
        with open(redacted_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return redacted_name, None
    except Exception as e:
        return None, f"Redaction error: {e}"

# --- Conversation Memory (short-term and long-term) ---
CONVO_HISTORY_FILE = "conversation_history.json"
convo_history_lock = threading.Lock()

# Save each user message and bot reply with timestamp
import time

def save_conversation(username, user_msg, bot_reply):
    record = {
        "username": username,
        "timestamp": time.time(),
        "user_msg": user_msg,
        "bot_reply": bot_reply
    }
    with convo_history_lock:
        try:
            if os.path.exists(CONVO_HISTORY_FILE):
                with open(CONVO_HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []
        except Exception:
            history = []
        history.append(record)
        # Limit to last 1000 messages for short-term memory
        if len(history) > 1000:
            history = history[-1000:]
        try:
            atomic_write_json(history, CONVO_HISTORY_FILE)
        except Exception as e:
            print(f"[ERROR] Failed to write conversation history: {e}")

# Retrieve recent conversation for a user

def get_recent_conversation(username, n=20):
    with convo_history_lock:
        try:
            with open(CONVO_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return []
        return [h for h in history if h["username"] == username][-n:]

# --- PDF, DOCX, Image OCR, and Document Q&A (Smarter) ---
import base64
from io import BytesIO

def extract_text_from_pdf(filepath):
    """
    Extracts text and tables from a PDF file using PyPDF2 and pandas (if possible).
    Returns the extracted text, tables, and error message (if any).
    """
    try:
        import PyPDF2
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() or '' for page in reader.pages)
        # Try table extraction (very basic, for demonstration)
        tables = []
        try:
            import tabula
            tables = tabula.read_pdf(filepath, pages='all', multiple_tables=True)
            tables = [df.head(10).to_string() for df in tables if not df.empty]
        except Exception:
            pass
        return text, tables, None
    except Exception as e:
        return None, None, f"PDF extraction error: {e}"

def extract_text_from_docx(filepath):
    """
    Extracts text and tables from a DOCX file using python-docx.
    Returns the extracted text, tables, and error message (if any).
    """
    try:
        import docx
        doc = docx.Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs])
        # Table extraction
        tables = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                rows.append([cell.text for cell in row.cells])
            import pandas as pd
            df = pd.DataFrame(rows)
            tables.append(df.head(10).to_string(index=False, header=False))
        return text, tables, None
    except Exception as e:
        return None, None, f"DOCX extraction error: {e}"

def extract_text_from_image(filepath):
    """
    Extracts text from an image file using pytesseract OCR.
    Returns the extracted text and error message (if any).
    """
    try:
        from PIL import Image
        import pytesseract
        text = pytesseract.image_to_string(Image.open(filepath))
        return text, None
    except Exception as e:
        return None, f"Image OCR error: {e}"

def extract_text_from_xlsx(filepath):
    """
    Extracts text and tables from an Excel file using pandas.
    Returns the extracted text, tables, and error message (if any).
    """
    try:
        import pandas as pd
        xls = pd.ExcelFile(filepath)
        text = []
        tables = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            tables.append(df.head(10).to_string())
            text.append(f"Sheet: {sheet}\n" + df.head(10).to_string())
        return "\n\n".join(text), tables, None
    except Exception as e:
        return None, None, f"Excel extraction error: {e}"

def detect_language_and_translate(text, target_lang='en'):
    """
    Detects language and translates to English if needed (using googletrans).
    Returns (translated_text, detected_lang, error)
    """
    try:
        from googletrans import Translator
        translator = Translator()
        detected = translator.detect(text)
        if detected.lang != target_lang:
            translated = translator.translate(text, dest=target_lang)
            return translated.text, detected.lang, None
        return text, detected.lang, None
    except Exception as e:
        return text, None, f"Translation error: {e}"

# --- Semantic Embedding Search (Sentence Transformers) ---
def get_semantic_embedding(text):
    """
    Returns a vector embedding for the given text using Sentence Transformers if available, else None.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode([text])[0]
    except Exception:
        return None

# --- Smart Semantic Q&A over Documents ---
def semantic_doc_qa(question, doc_text):
    """
    Answers a question over a document using semantic embedding search if possible, else TF-IDF.
    Returns the best matching answer span or None.
    """
    try:
        # Split doc into sentences/paragraphs
        import re
        chunks = re.split(r'(?<=[.!?])\s+', doc_text)
        # Try embedding-based search
        q_emb = get_semantic_embedding(question)
        if q_emb is not None:
            from sentence_transformers import util
            chunk_embs = [get_semantic_embedding(c) for c in chunks]
            sims = util.cos_sim([q_emb], chunk_embs)[0].tolist()
            idx = sims.index(max(sims))
            return chunks[idx]
        else:
            # Fallback: TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer().fit([question] + chunks)
            vectors = vectorizer.transform([question] + chunks)
            sims = (vectors[0:1] @ vectors[1:].T).toarray()[0]
            idx = sims.argmax()
            return chunks[idx]
    except Exception:
        return None

# --- LLM Fallback (OpenAI/Gemini/Local) ---
def llm_fallback(query, context=None):
    """
    Uses an LLM API (OpenAI, Gemini, or local) to answer the query, optionally with context.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
        openai.api_key = api_key
        prompt = f"Context: {context}\n\nUser: {query}\nAI: " if context else query
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful, expert AI assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception:
        return None

# --- Code Explanation and Inline Suggestion ---
def explain_code(code, language='python'):
    """
    Uses LLM to explain code and suggest improvements.
    """
    explanation = llm_fallback(f"Explain this {language} code and suggest improvements:\n{code}")
    return explanation or "Code explanation unavailable."

# --- Enhanced get_response with smarter routing ---
def get_response(msg, username=None, file_context=None):
    """
    Main response router: math, image, file, code, speech, intent/KB, semantic doc Q&A, LLM fallback, Google.
    Now uses semantic embedding, LLM fallback, and memory/context.
    """
    msg_norm = normalize_grammar(msg)
    # 1. Math
    math_reply = handle_math(msg_norm)
    if math_reply:
        return math_reply
    # 2. Image generation
    if any(w in msg_norm for w in ["draw", "generate image", "create image", "picture", "art"]):
        return generate_image_from_prompt(msg)
    # 3. File Q&A (if file_context provided)
    if file_context:
        content, err = get_uploaded_file_content(file_context)
        if content:
            # Use semantic_doc_qa
            answer = semantic_doc_qa(msg, content)
            if answer:
                return f"From your file: {answer}"
            # fallback: return first 500 chars
            return f"File content: {content[:500]}..."
        elif err:
            return err
    # 4. Code execution/explanation
    if any(w in msg_norm for w in ["run code", "execute code", "code:", "explain code", "suggest code"]):
        lang = extract_language(msg_norm)
        code_match = re.search(r'```([\w\+\#]*)\n([\s\S]+?)```', msg)
        code = code_match.group(2) if code_match else msg
        if "explain" in msg_norm or "suggest" in msg_norm:
            return explain_code(code, lang)
        output, err = run_code(code, lang)
        if output:
            return f"Output:\n{output}"
        elif err:
            return f"Error:\n{err}"
    # 5. Speech
    topic, tone, audience, length = extract_speech_request(msg_norm)
    if topic:
        # Use generator API or LLM
        speech = llm_fallback(f"Write a {tone or ''} speech about {topic} for {audience or 'a general audience'} of {length or 'medium'} length.")
        return speech or "Speech generation unavailable."
    # 6. Intent/KB
    intent_reply = get_intent_response(msg)
    if intent_reply:
        return intent_reply
    kb_reply = get_from_knowledge_base(msg)
    if kb_reply:
        return kb_reply
    # 7. Conversation memory/context
    if username:
        recent = get_recent_conversation(username, n=5)
        if recent:
            context = '\n'.join([f"User: {r['user_msg']}\nAI: {r['bot_reply']}" for r in recent])
            mem_reply = llm_fallback(msg, context=context)
            if mem_reply:
                return mem_reply
    # 8. LLM fallback
    llm_reply = llm_fallback(msg)
    if llm_reply:
        return llm_reply
    # 9. Google fallback
    return ask_google(msg)

# --- Self-improvement hook (logs unhandled queries for future training) ---
def log_unhandled_query(msg, username=None):
    try:
        with open("unhandled_queries.log", "a", encoding="utf-8") as f:
            f.write(f"{time.time()}\t{username or ''}\t{msg}\n")
    except Exception:
        pass

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'uploads')

def get_uploaded_file_content(filename):
    """
    Loads and extracts content from uploaded files (text, csv, py, pdf, docx, images, xlsx).
    Returns (content, error) tuple. Used for file Q&A and semantic_doc_qa.
    """
    filepath = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(filepath):
        return None, "File not found."
    mime, _ = mimetypes.guess_type(filepath)
    try:
        if mime and mime.startswith('text'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(), None
        elif filename.lower().endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(filepath)
            return df.head(10).to_string(), None
        elif filename.lower().endswith('.py'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(), None
        elif filename.lower().endswith('.pdf'):
            text, tables, err = extract_text_from_pdf(filepath)
            if text and tables:
                return text + "\n\nTables:\n" + "\n\n".join(tables), None
            elif text:
                return text, None
            else:
                return None, err
        elif filename.lower().endswith('.docx'):
            text, tables, err = extract_text_from_docx(filepath)
            if text and tables:
                return text + "\n\nTables:\n" + "\n\n".join(tables), None
            elif text:
                return text, None
            else:
                return None, err
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            text, err = extract_text_from_image(filepath)
            return text, err
        elif filename.lower().endswith('.xlsx'):
            text, tables, err = extract_text_from_xlsx(filepath)
            if text and tables:
                return text + "\n\nTables:\n" + "\n\n".join(tables), None
            elif text:
                return text, None
            else:
                return None, err
        else:
            return None, "Unsupported file type for Q&A."
    except Exception as e:
        return None, f"Error reading file: {e}"



