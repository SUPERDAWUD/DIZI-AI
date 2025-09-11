from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from chat import get_response, web_summarize_url, web_search_and_summarize, build_vector_index, query_vector_index
from utils import start_generator_api
from functools import wraps
import json
from dotenv import load_dotenv
import time

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'json', 'py', 'zip', 'docx', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEV_MODE_PASSWORD = os.getenv('DEV_MODE_PASSWORD', 'supersecret')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'devsecret')

# Dev Mode decorator

def dev_mode_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('dev_mode'):
            return redirect(url_for('dev'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_route():
    user_msg = request.json.get("message")
    user_name = request.json.get("user")
    bot_reply = get_response(user_msg, username=user_name, use_web=bool(request.json.get('web')), use_rag=bool(request.json.get('rag')))
    return jsonify({"response": bot_reply})

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"success": True, "filename": filename})
    return jsonify({"error": "File type not allowed"}), 400

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/dev', methods=['GET', 'POST'])
def dev():
    # If already logged in, always show dev panel
    if session.get('dev_mode'):
        return render_template('dev_panel.html')
    # Otherwise, show login form
    error = None
    if request.method == 'POST':
        password = request.form.get('password')
        if password == DEV_MODE_PASSWORD:
            session['dev_mode'] = True
            return render_template('dev_panel.html')
        error = 'Incorrect password'
    return render_template('dev_login.html', error=error)

# Optional: API for chat sidebar
@app.route('/api/chats')
def api_chats():
    try:
        CONVO_HISTORY_FILE = "conversation_history.json"
        if os.path.exists(CONVO_HISTORY_FILE):
            with open(CONVO_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        chats = []
        for i, entry in enumerate(history[-50:]):
            chats.append({
                'id': i,
                'title': entry.get('user_msg', '')[:30] or f'Chat {i+1}',
                'timestamp': entry.get('timestamp', '')
            })
        return jsonify(chats)
    except Exception as e:
        return jsonify([])

# --- Gemini Models Page ---
@app.route('/dev/gemini_models')
@dev_mode_required
def dev_gemini_models():
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        return render_template('gemini_models.html', error='GEMINI_API_KEY not set in environment.', models=[])
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        models = genai.list_models()
        model_list = []
        for m in models:
            model_list.append({
                'name': getattr(m, 'name', str(m)),
                'supported_generation_methods': getattr(m, 'supported_generation_methods', [])
            })
        return render_template('gemini_models.html', models=model_list)
    except Exception as e:
        return render_template('gemini_models.html', error=str(e), models=[])

# --- Windows-compatible Math Handler ---
import concurrent.futures

def handle_math(msg):
    import sympy, re
    from sympy import symbols, Eq, solve, simplify, expand, factor, diff, integrate
    x, y = symbols("x y")
    original_msg = msg.strip()
    msg = original_msg.lower().replace("^", "**")
    math_prefixes = [
        "what is ", "whats ", "what's ", "calculate ", "solve ", "compute ", "evaluate ", "find ", "give me ", "tell me ", "can you solve ", "can you calculate ", "can you evaluate ", "can you compute "
    ]
    for prefix in math_prefixes:
        if msg.startswith(prefix):
            msg = msg[len(prefix):].strip()
            break
    def math_eval(expr):
        try:
            if re.search(r"[\+\-\*/]", expr) and not re.fullmatch(r"\d+", expr):
                result = sympy.sympify(expr).evalf()
                if result == int(result):
                    result = int(result)
                return f"The answer is {result}"
            return None
        except Exception as e:
            return f"Math error: {e}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(math_eval, msg)
        try:
            return future.result(timeout=2)
        except concurrent.futures.TimeoutError:
            return "Math error: Calculation took too long. Please try a simpler question."

# --- API: List repo files ---
@app.route('/api/code-assistant-files')
@dev_mode_required
def code_assistant_files():
    repo_dir = os.getcwd()
    files = []
    for root, dirs, filenames in os.walk(repo_dir):
        for fname in filenames:
            # Only show code/text files
            if fname.endswith(('.py', '.js', '.ts', '.html', '.css', '.md', '.json', '.txt')):
                files.append(os.path.relpath(os.path.join(root, fname), repo_dir))
    return jsonify({'files': files})

# --- API: Read file ---
@app.route('/api/code-assistant-read', methods=['POST'])
@dev_mode_required
def code_assistant_read():
    data = request.get_json()
    path = data.get('path')
    repo_dir = os.getcwd()
    abs_path = os.path.abspath(os.path.join(repo_dir, path))
    if not abs_path.startswith(repo_dir):
        return jsonify({'error': 'Invalid path.'})
    try:
        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)})

# --- API: Write file ---
@app.route('/api/code-assistant-write', methods=['POST'])
@dev_mode_required
def code_assistant_write():
    data = request.get_json()
    path = data.get('path')
    content = data.get('content', '')
    repo_dir = os.getcwd()
    abs_path = os.path.abspath(os.path.join(repo_dir, path))
    if not abs_path.startswith(repo_dir):
        return jsonify({'error': 'Invalid path.'})
    try:
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/save_chat', methods=['POST'])
def api_save_chat():
    data = request.get_json()
    username = data.get('user')
    chat = data.get('chat')
    if not username or not chat:
        return jsonify({'success': False, 'error': 'Missing user or chat.'})
    # Save chat per user
    chat_file = os.path.join('user_chats', f'{username}.json')
    os.makedirs('user_chats', exist_ok=True)
    try:
        if os.path.exists(chat_file):
            with open(chat_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        history.append({'chat': chat, 'timestamp': time.time()})
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load_chat', methods=['POST'])
def api_load_chat():
    data = request.get_json()
    username = data.get('user')
    if not username:
        return jsonify({'success': False, 'error': 'Missing user.'})
    chat_file = os.path.join('user_chats', f'{username}.json')
    if not os.path.exists(chat_file):
        return jsonify({'success': False, 'error': 'No chat found.'})
    with open(chat_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    return jsonify({'success': True, 'history': history})

if __name__ == "__main__":
    start_generator_api()
    start_indexer_watch()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



@app.route('/api/web-summarize', methods=['POST'])
def api_web_summarize():
    data = request.get_json() or {}
    query = data.get('query')
    url = data.get('url')
    n = int(data.get('num_results', 3))
    if url:
        result = web_summarize_url(url, question=query)
        return jsonify({'ok': True, 'result': result})
    if query:
        result = web_search_and_summarize(query, num_results=n)
        return jsonify({'ok': True, 'result': result})
    return jsonify({'ok': False, 'error': 'Provide url or query'}), 400


@app.route('/api/rag/reindex', methods=['POST'])
@dev_mode_required
def api_rag_reindex():
    ok, msg = build_vector_index()
    return jsonify({'ok': ok, 'message': msg})

@app.route('/api/rag/query', methods=['POST'])
@dev_mode_required
def api_rag_query():
    data = request.get_json() or {}
    q = data.get('query','')
    k = int(data.get('top_k', 5))
    hits = query_vector_index(q, top_k=k)
    return jsonify({'ok': True, 'hits': hits})

@app.route('/api/set-device', methods=['POST'])
@dev_mode_required
def api_set_device():
    from chat import local_gpt
    data = request.get_json() or {}
    device = data.get('device','auto')
    try:
        if hasattr(local_gpt, 'set_device'):
            local_gpt.set_device(device)
    except Exception:
        pass
    return jsonify({'device': device})

@app.route('/api/set-aggregation-mode', methods=['POST'])
@dev_mode_required
def api_set_aggregation_mode():
    import chat as chat_mod
    data = request.get_json() or {}
    mode = (data.get('mode') or 'parallel').lower()
    if mode not in ('parallel','priority'):
        return jsonify({'ok': False, 'error': 'mode must be parallel or priority'}), 400
    chat_mod.AGGREGATION_MODE = mode
    return jsonify({'ok': True, 'mode': chat_mod.AGGREGATION_MODE})

@app.route("/chat/stream", methods=["POST"])
def chat_route_stream():
    """Server-Sent Events streaming of a chat response (chunked or Gemini streaming)."""
    def gen():
        try:
            data = request.get_json() or {}
            user_msg = data.get("message", "")
            user_name = data.get("user", None)
            prefer_gemini = bool(data.get("prefer_gemini_stream", True))
            web = bool(data.get("web", False))
            rag = bool(data.get("rag", False))
            # Try Gemini token streaming if configured
            if prefer_gemini:
                try:
                    import os
                    import google.generativeai as genai
                    api_key = os.getenv('GEMINI_API_KEY')
                    if api_key:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('models/gemini-2.5-pro')
                        resp = model.generate_content(user_msg, stream=True)
                        # Emit meta
                        yield "data: " + json.dumps({"meta": {"type":"gemini_stream"}}) + "\n\n"
                        for chunk in resp:
                            if hasattr(chunk, 'text') and chunk.text:
                                yield "data: " + json.dumps({"delta": chunk.text}) + "\n\n"
                        yield "data: " + json.dumps({"done": True}) + "\n\n"
                        return
                except Exception:
                    pass
            # Fallback: use full response then chunk it
            reply = get_response(user_msg, username=user_name, all_files=False, file_context=None, use_web=web, use_rag=rag)
            meta = {k: reply.get(k) for k in ("type", "language", "personality", "followup") if isinstance(reply, dict)}
            yield "data: " + json.dumps({"meta": meta}) + "\n\n"
            content = reply.get("content", "") if isinstance(reply, dict) else str(reply)
            buf = []
            for token in content.split():
                buf.append(token)
                if len(" ".join(buf)) > 48:
                    chunk = " ".join(buf) + " "
                    yield "data: " + json.dumps({"delta": chunk}) + "\n\n"
                    buf = []
            if buf:
                chunk = " ".join(buf)
                yield "data: " + json.dumps({"delta": chunk}) + "\n\n"
            yield "data: " + json.dumps({"done": True}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
    return app.response_class(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
# --- Background Vector Indexer ---
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:
    Observer = None
    FileSystemEventHandler = object

class _IndexHandler(FileSystemEventHandler):
    def __init__(self):
        self._timer = None
    def on_any_event(self, event):
        try:
            import threading
            if self._timer:
                self._timer.cancel()
            def _rebuild():
                try:
                    from chat import build_vector_index
                    build_vector_index()
                except Exception:
                    pass
            self._timer = threading.Timer(1.0, _rebuild)
            self._timer.start()
        except Exception:
            pass

def start_indexer_watch():
    if Observer is None:
        return
    try:
        import os
        handler = _IndexHandler()
        obs = Observer()
        up = os.path.join(os.getcwd(), 'uploads')
        dt = os.path.join(os.getcwd(), 'data')
        if os.path.exists(up):
            obs.schedule(handler, up, recursive=True)
        if os.path.exists(dt):
            obs.schedule(handler, dt, recursive=True)
        obs.daemon = True
        obs.start()
    except Exception:
        pass



from gtts import gTTS

@app.route('/api/tts', methods=['POST'])
def api_tts():
    data = request.get_json() or {}
    text = data.get('text','')
    lang = data.get('lang','en')
    if not text.strip():
        return jsonify({'ok': False, 'error': 'Missing text'}), 400
    try:
        outdir = os.path.join('static','generated')
        os.makedirs(outdir, exist_ok=True)
        fname = f"tts_{int(time.time()*1000)}.mp3"
        fpath = os.path.join(outdir, fname)
        tts = gTTS(text=text, lang=lang)
        tts.save(fpath)
        return jsonify({'ok': True, 'url': f"/static/generated/{fname}"})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500






@app.route('/api/image/generate', methods=['POST'])
def api_image_generate():
    from chat import generate_image_advanced
    data = request.get_json() or {}
    prompt = data.get('prompt','')
    negative = data.get('negative_prompt')
    seed = data.get('seed')
    n = data.get('num_images', 1)
    if not prompt:
        return jsonify({'ok': False, 'error': 'Missing prompt'}), 400
    urls = generate_image_advanced(prompt, negative_prompt=negative, seed=seed, num_images=n)
    return jsonify({'ok': True, 'urls': urls})
