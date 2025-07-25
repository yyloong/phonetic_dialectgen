from openai import OpenAI
import toml

config = toml.load("2025.07.25_config.toml")
api_key = config["moonshot"]["api_key"]
base_url = config["moonshot"]["base_url"]

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

def preprocess_tts(sentence):
    content = f"""你的任务是将以下给出的中文句子进行汉语TTS的文本规范化:
    1. 进行汉语中数字与数量表达的规范化；
    2. 数学、物理等符号的口语化转换；
    3. 将所有非中文文本直接进行汉语翻译，使得转换后的句子中仅包含中文；
    4. 不要对原句子进行任何其他修改，保留标点符号和中文部分。
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


print(preprocess_tts("1989年是哪位明星的出生年份？"))