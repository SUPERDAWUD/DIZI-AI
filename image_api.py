from flask import Flask, request, jsonify
from datetime import datetime
import os
import shutil

app = Flask(__name__)
OUT_DIR = "static/generated"
os.makedirs(OUT_DIR, exist_ok=True)

@app.route("/generate/image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "mystery object")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{prompt.replace(' ', '_')}_{timestamp}.jpg"
    output_path = os.path.join(OUT_DIR, filename)

    # TEMP: copy placeholder image
    shutil.copy("static/placeholder.jpg", output_path)

    return jsonify({ "image_url": f"/{output_path}" })

if __name__ == "__main__":
    app.run(port=5050)
