from flask import Flask, render_template, request, jsonify
from chat import get_response
from utils import start_generator_api

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_route():
    user_msg = request.json.get("message")
    user_name = request.json.get("user")
    bot_reply = get_response(user_msg, username=user_name)
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    import os
    start_generator_api()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

