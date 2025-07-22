import json
import os


def convert_sentence_to_codes(sentence):
    # 定义中文标点符号集合
    chinese_punctuation = "。！？，、；：“”‘’（）【】《》"

    # 加载映射字典
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_file = os.path.join(script_dir, "mapping_dict.json")

    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping_dict = json.load(f)
    except FileNotFoundError:
        print("映射文件未找到")
        return ()
    except json.JSONDecodeError:
        print("映射文件格式错误")
        return ()

    codes = []  # 存储编码结果

    # 遍历每个字符
    for char in sentence:
        # 尝试在字典中查找字符
        if char in mapping_dict:
            codes.append(mapping_dict[char])
        else:
            # 检查是否为中文标点
            if char in chinese_punctuation:
                continue  # 忽略标点
            else:
                return ()  # 非标点字符缺失映射

    return tuple(codes)


# 示例用法
if __name__ == "__main__":
    test_sentence = "ni35 xuŋ35 y215 ʂuei215 tɕin51 thou51 lɤ0 ʂu51 tɕy51 xuŋ35 liou35 tɤ0 tɕie55 tau51 。"
    result = convert_sentence_to_codes(test_sentence)
    print(result)
