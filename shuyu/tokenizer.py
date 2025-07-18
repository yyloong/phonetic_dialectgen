import torch

# IPA 字符串中的字符总数: 46 这个值记为 n

class ShuTokenizer:
    def __init__(self):
        self.mapping = {
            " ": 1,
            "1": 2,
            "2": 3,
            "3": 4,
            "4": 5,
            "5": 6,
            "a": 7,
            "b": 8,
            "c": 9,
            "d": 10,
            "e": 11,
            "f": 12,
            "g": 13,
            "h": 14,
            "i": 15,
            "j": 16,
            "k": 17,
            "l": 18,
            "m": 19,
            "n": 20,
            "o": 21,
            "p": 22,
            "q": 23,
            "r": 24,
            "s": 25,
            "t": 26,
            "u": 27,
            "v": 28,
            "w": 29,
            "x": 30,
            "y": 31,
            "z": 32,
            '。': 33,
            '.': 33,  # '.' 和 '。' 都映射到同一个 ID
            '，': 34,
            ',': 34,
            '、': 34,  # '、', '，' 和 ',' 都映射到同一个 ID
            '！': 35,
            '!': 35,  # '!' 和 '！' 都映射到同一个 ID
            '：': 36,
            ':': 36,  # ':' 和 '：' 都映射到同一个 ID
            '；': 37,
            ';': 37,  # ';' 和 '；' 都映射到同一个 ID
            '？': 38,
            '?': 38  # '?' 和 '？' 都映射到同一个 ID
        }

    def __call__(self, text):
        # 定义中文标点符号集合
        punctuations = "“”‘’（）【】《》\"'()[]{}<>《》"

        codes = []  # 存储编码结果

        # 遍历每个字符
        for char in text:
            # 尝试在字典中查找字符
            if char in self.mapping:
                codes.append(self.mapping[char])
            else:
                # 检查是否为中文标点
                if char in punctuations:
                    continue  # 忽略标点
                else:
                    raise ValueError(f"字符 '{char}' 缺失映射")

        return torch.tensor(codes, dtype=torch.long)

if __name__ == "__main__":
    # Example usage
    print("Testing TTSTokenizer...")
    text = """ye51 liaŋ51 thou55 thou55 kei215 xai215 ou55 suŋ51 tɕhy51 niŋ35 məŋ35 khou215 uei51 tɤ0 thaŋ35 kuo215 。"""
    print(len(text))
    print(f"Tokenizing ...")
    tokenizer = ShuTokenizer()
    tokens = tokenizer(text)
    print(tokens)  # Should print a tensor of token IDs
    print(tokens.shape)  # Should match the length of the input text