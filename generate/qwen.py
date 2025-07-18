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

text = "那我来给大家推荐一款T恤，这款呢真的是超级好看，这个颜色呢很显气质，而且呢也是搭配的绝佳单品，大家可以闭眼入，真的是非常好看，对身材的包容性也很好，不管啥身材的宝宝呢，穿上去都是很好看的。推荐宝宝们下单哦。"
response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
    model="qwen-tts-latest",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    text=text,
    voice="Dylan",
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