

 
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from chat import get_response
from utils import start_generator_api
from functools import wraps
import json
from dotenv import load_dotenv

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
    bot_reply = get_response(user_msg, username=user_name)
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

if __name__ == "__main__":
    start_generator_api()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

