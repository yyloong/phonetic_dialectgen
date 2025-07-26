import json
import time
import toml

from tqdm import tqdm

from openai import OpenAI

config = toml.load("/home/u-shigc/LLM/2025.07.09_config.toml")
api_key = config["moonshot"]["api_key"]
base_url = config["moonshot"]["base_url"]

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)


def preprocess_tts(sentence):
    content = """你的任务是将用户输入的中文句子进行汉语TTS的text normalization:
    1. 进行汉语中数字与数量表达的规范化；
    2. 数学、物理等符号的口语化转换；
    3. 将所有非中文文本直接进行汉语翻译，使得转换后的句子中仅包含中文；
    注意：输出的结果中不能出现英文或阿拉伯数字！不要输出任何的额外信息！
    """
    response = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": sentence},
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content


print(preprocess_tts("我有1个苹果和2个香蕉。"))


from pathlib import Path

# 诸序.pdf 是一个示例文件, 我们支持 pdf, doc 以及图片等格式, 对于图片和 pdf 文件，提供 ocr 相关能力
file_object = client.files.create(
    file=Path("/home/u-shigc/LLM/诸论.pdf"), purpose="file-extract"
)

# 获取结果
# file_content = client.files.retrieve_content(file_id=file_object.id)
# 注意，之前 retrieve_content api 在最新版本标记了 warning, 可以用下面这行代替
# 如果是旧版本，可以用 retrieve_content
file_content = client.files.content(file_id=file_object.id).text

# 把它放进请求中
messages = [
    {
        "role": "system",
        "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
    },
    {
        "role": "system",
        "content": file_content,
    },
    {
        "role": "user",
        "content": "提取文件中的文字（若总字数超过100字，则只提取前100个字即可）,仅能使用中文汉字回答（需要将所有英文字母、数字等转为汉字，可以直接采用音译法）。仅需输出一段话，不需要其他内容。每句话之间用空格分隔，保留标点符号。",
    },
]

# 然后调用 chat-completion, 获取 Kimi 的回答
completion = client.chat.completions.create(
    model="kimi-k2-0711-preview",
    messages=messages,
    temperature=0.6,
)

print(completion.choices[0].message.content)
