from flask import Flask, render_template, request, jsonify
import os

# Reuse translation helpers from chat.py
from chat import translate_text, detect_language, ISO_LANGS

app = Flask(__name__)

@app.route('/')
def translator_home():
    langs = sorted(ISO_LANGS.items())
    return render_template('translator.html', langs=langs)

@app.route('/translate', methods=['POST'])
def translator_translate():
    data = request.get_json() or {}
    text = data.get('text','')
    target = (data.get('target') or 'en').lower()
    if not text.strip():
        return jsonify({'ok': False, 'error': 'Missing text'}), 400
    src = detect_language(text)
    translated = translate_text(text, target_lang=target)
    return jsonify({'ok': True, 'source': src, 'target': target, 'translated': translated})

if __name__ == '__main__':
    # Chrome blocks some ports (e.g., 6000/X11) as unsafe.
    # Use a safe default like 8000 unless overridden by TRANSLATOR_PORT.
    port = int(os.environ.get('TRANSLATOR_PORT', '8000'))
    app.run(host='0.0.0.0', port=port)
