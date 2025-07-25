from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
from synthesize import synthesize

app = Flask(__name__)
CORS(app)

@ app.route('/')
def index():
    return send_file('index.html')

@app.route('/tts', methods=['POST'])
def generate():
    # 读取请求数据
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400
    language = data.get("language", "")
    if language not in ["pinyin", "jyutping"]:
        return jsonify({"error": "Invalid language"}), 400
    # 合成语音
    model_mapping = {
        "pinyin": "./weights/mandarin.pth",
        "jyutping": "./weights/cantonese.pth"
    }
    checkpoint_path = model_mapping[language]
    synthesize(checkpoint_path, text, language)
    wav_path = "output.wav"
    if not os.path.exists(wav_path):
        return jsonify({"error": "Audio file not found"}), 500
    return send_file(wav_path, mimetype="audio/wav")

# flask --app web run --host=0.0.0.0
# gunicorn -w 4 -b 0.0.0.0:5000 web:app