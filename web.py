import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from synthesize import synthesize
from shuyu.synthesize import synthesize_sichuan

from openai import OpenAI
import toml

import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

config = toml.load("/home/u-wuhc/phonetic_dialectgen/tmp/2025.07.25_config.toml")
api_key = config["moonshot"]["api_key"]
base_url = config["moonshot"]["base_url"]

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

app = Flask(__name__)
CORS(app)

def preprocess_tts(sentence):
    content = f"""你的任务是将以下给出的中文句子进行汉语TTS的文本规范化:
    1. 去除所有特殊符号，保留标点符号；
    2. 将数字，年份等表达的规范化为汉语（年份中的0请用‘零’）；
    3. 数学、物理等符号的口语化转换；
    4. 将所有非中文文本（包含字母缩写名称）直接进行汉语翻译，使得转换后的句子中仅包含中文；
    5. 不要对原句子进行任何其他修改和润色。
    注意（重要）：输出的结果中不能出现英文或阿拉伯数字！不要输出任何的额外信息！
    
    文本内容如下：
    {sentence}
    """
    response = client.chat.completions.create(
        model="kimi-latest",
        messages=[
            {"role": "user", "content": content},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content

def clean(text):
    """清理markdown文本"""
    lines = text.splitlines()
    cleaned_lines = []
    start = False
    for line in lines:
        if len(line) >= 2 and line[0] == "#" and line[1] == " ":
            cleaned_lines.append(line[2:])
        if line == "正在加载":
            start = True
            continue
        if len(line) >= 3 and line[0] == "编" and line[1] == "辑" and line[2] == "：":
            start = False
            continue
        if start:
            if not line.startswith("![]("):
                cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

async def crawl_web(url):
    browser_conf = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
    )
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config
        )
        content = clean(result.markdown)
        print(f"Processed: {result.url}")
        return content

@app.route('/')
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
    if language not in ["pinyin", "jyutping", "shupin"]:
        return jsonify({"error": "Invalid language"}), 400
    # 合成语音
    model_mapping = {
        "pinyin": "./weights/mandarin.pth",
        "jyutping": "./weights/cantonese.pth",
        "shupin": "./shuyu/weights/sichuan.pth"
    }
    checkpoint_path = model_mapping[language]
    # 规范化文本（如果需要）
    # 检测是否包含数字或英文字幕
    if any(char.isdigit() for char in text) or any(char.isalpha() for char in text):
        text = preprocess_tts(text)
    print(f"🎤 规范化后的文本: {text}")
    if language == "shupin":
        synthesize_sichuan(checkpoint_path, text)
    else:
        synthesize(checkpoint_path, text, language)
    wav_path = "output.wav"
    if not os.path.exists(wav_path):
        return jsonify({"error": "Audio file not found"}), 500
    return send_file(wav_path, mimetype="audio/wav")

@app.route('/web_reader', methods=['POST'])
def web_reader():
    data = request.get_json()
    language = data.get("language", "")
    if language not in ["pinyin", "jyutping", "shupin"]:
        return jsonify({"error": "Invalid language"}), 400
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "URL is required"}), 400
    text = asyncio.run(crawl_web(url))
    # 合成语音
    model_mapping = {
        "pinyin": "./weights/mandarin.pth",
        "jyutping": "./weights/cantonese.pth",
        "shupin": "./shuyu/weights/sichuan.pth"
    }
    checkpoint_path = model_mapping[language]
    # 规范化文本（如果需要）
    # 检测是否包含数字或英文字幕
    if any(char.isdigit() for char in text) or any(char.isalpha() for char in text):
        text = preprocess_tts(text)
    print(f"🎤 规范化后的文本: {text}")
    if language == "shupin":
        synthesize_sichuan(checkpoint_path, text)
    else:
        synthesize(checkpoint_path, text, language)
    wav_path = "output.wav"
    if not os.path.exists(wav_path):
        return jsonify({"error": "Audio file not found"}), 500
    return send_file(wav_path, mimetype="audio/wav")

@app.route('/file_reader', methods=['POST'])
def file_reader():
    file = request.files.get('file')
    if not file or not file.filename.endswith('.txt'):
        return jsonify({"error": "Invalid file"}), 400
    text = file.read().decode('utf-8')
    if not text:
        return jsonify({"error": "File is empty"}), 400
    language = request.form.get("language", "")
    if language not in ["pinyin", "jyutping", "shupin"]:
        return jsonify({"error": "Invalid language"}), 400
    # 合成语音
    model_mapping = {
        "pinyin": "./weights/mandarin.pth",
        "jyutping": "./weights/cantonese.pth",
        "shupin": "./shuyu/weights/sichuan.pth"
    }
    checkpoint_path = model_mapping[language]
    # 规范化文本（如果需要）
    # 检测是否包含数字或英文字幕
    if any(char.isdigit() for char in text) or any(char.isalpha() for char in text):
        text = preprocess_tts(text)
    print(f"🎤 规范化后的文本: {text}")
    if language == "shupin":
        synthesize_sichuan(checkpoint_path, text)
    else:
        synthesize(checkpoint_path, text, language)
    wav_path = "output.wav"
    if not os.path.exists(wav_path):
        return jsonify({"error": "Audio file not found"}), 500
    return send_file(wav_path, mimetype="audio/wav")

# flask --app web run --host=0.0.0.0
# gunicorn -w 4 -b 0.0.0.0:5000 web:app