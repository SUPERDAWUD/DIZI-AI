from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from chat import get_response
from utils import start_generator_api
from functools import wraps
import json

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'json', 'py', 'zip', 'docx', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEV_MODE_PASSWORD = os.getenv('DEV_MODE_PASSWORD', 'supersecret')

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

if __name__ == "__main__":
    start_generator_api()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

