import os
import pandas as pd
import requests
import tqdm
# pip install dashscope
import dashscope
# pip install python-dotenv
from dotenv import load_dotenv

# Ensure you have the DASHSCOPE_API_KEY set in your environment variables
# 在当前目录下创建一个 .env 文件，并添加以下内容：
# DASHSCOPE_API_KEY=your_api_key_here
load_dotenv()

df = pd.read_csv("aitts3_shu.csv", encoding='utf-8')

print(len(df))

for i in tqdm.tqdm(range(7000, 7010)):
    text = df.loc[i, 'text']
    name = df.loc[i, 'audio']
    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
        model="qwen-tts-latest",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        text=text,
        voice="Sunny", 
    )
    audio_url = response.output.audio["url"]
    save_path = f"sichuan/{name}.wav"  # 自定义保存路径

    response = requests.get(audio_url)
    response.raise_for_status()  # 检查请求是否成功
    with open(save_path, 'wb') as f:
        f.write(response.content)
    # print(f"音频文件已保存至：{save_path}")