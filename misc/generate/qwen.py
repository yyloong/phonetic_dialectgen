import os
import requests
# pip install dashscope
import dashscope
# pip install python-dotenv
from dotenv import load_dotenv

# Ensure you have the DASHSCOPE_API_KEY set in your environment variables
# 在当前目录下创建一个 .env 文件，并添加以下内容：
# DASHSCOPE_API_KEY=your_api_key_here
load_dotenv()

text = "萤火虫在回忆中跳舞，诉说着古老的故事。"
response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
    model="qwen-tts-latest",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    text=text,
    voice="Sunny",
)
# voice: 
# Dylan=北京话
# Jada=吴语
# Sunny=四川话
audio_url = response.output.audio["url"]
save_path = "downloaded_audio.wav"  # 自定义保存路径

try:
    response = requests.get(audio_url)
    response.raise_for_status()  # 检查请求是否成功
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"音频文件已保存至：{save_path}")
except Exception as e:
    print(f"下载失败：{str(e)}")