import json
import torch
import os

def ipa_to_tensor(text):
    # 加载映射字典
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_file = os.path.join(script_dir, "mapping_dict.json")

    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping= json.load(f)
    except FileNotFoundError:
        print("映射文件未找到")
        return ()
    except json.JSONDecodeError:
        print("映射文件格式错误")
        return ()

    punctuations = "“”‘’（）【】《》\"'()[]{}<>《》"

    codes = []  # 存储编码结果

    # 遍历每个字符
    for char in text:
        # 尝试在字典中查找字符
        if char in mapping:
            codes.append(mapping[char])
        else:
            # 检查是否为中文标点
            if char in punctuations:
                continue  # 忽略标点
            else:
                raise ValueError(f"字符 '{char}' 缺失映射")

    return torch.tensor(codes, dtype=torch.long)

