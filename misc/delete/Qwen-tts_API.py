# DashScope SDK 版本不低于 1.23.1
import sys
import os
import tqdm
import dashscope
import requests
import pandas as pd
import multiprocess as mp


def get_Qwen_audio(text, save_path):
    print(os.getenv("DASHSCOPE_API_KEY"))
    if os.path.exists(save_path):
        return save_path
    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
        model="qwen-tts-latest",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        text=text,
        voice="Dylan",
    )
    print(response)
    try:
        audio_url = response.output.audio["url"]
        response = requests.get(audio_url)
        response.raise_for_status()  # 检查请求是否成功
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"音频文件已保存至：{save_path}")
    except Exception as e:
        print(f"{save_path}:下载失败：{str(e)}")

def one_process_Qwen_audio(text_path, save_dir):
    if not os.path.exists(text_path):
        print("text path error")
    elif not os.path.exists(save_dir):
        print("save_dir error")
    df = pd.read_csv(text_path)
    text = df['句子']
    text = list(text)
    save_paths = [f"{save_dir}/{i}.wav" for i in range(len(text))]
    for i in range(len(text)):
        get_Qwen_audio(text=text[i],save_path=save_paths[i])
'''
因为平台限流所以无法使用
def multi_get_Qwen_audio(text_path, save_dir, num_workers=2):
    if not os.path.exists(text_path):
        print("text path error")
    elif not os.path.exists(save_dir):
        print("save_dir error")
    df = pd.read_csv(text_path)
    text = df['句子']
    text = list(text)
    save_paths = [f"{save_dir}/{i}.wav" for i in range(len(text))]
    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    get_Qwen_audio,
                    zip(text, save_paths),
                ),
                total=len(text),
                desc="get audio",
            )
        )
    fail_list = []
    for i, result in enumerate(results):
        if result["status"] == "success":
            print(f"成功生成: {i} (文本: {result['text']})")
        else:
            fail_list.append(i)
            print(f"生成失败: {i}，错误: {result['message']}")

    print(f"fail list:{fail_list}")
    print(f"失败 {len(fail_list)} 条")
'''

one_process_Qwen_audio("../AItts3.csv", "../Qwen-tts_Beijing")
