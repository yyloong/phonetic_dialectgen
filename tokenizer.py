import torch

# IPA 字符串中的字符总数: 46 这个值记为 n

class TTSTokenizer:
    """
    GlowTTS Tokenizer: transforms IPA strings into token IDs.

    input: text (str) - IPA 字符串
    output: token_ids (torch.Tensor, dtype=torch.long) - 每个元素是 [1, n] 范围内的整数

    token_ids 的长度与输入字符串长度相同

    注：如果 n = 100, 则 token_ids 是 [0, 100] 范围内的整数，
    其中 0 要留出，是因为作为 padding 以填充数据集以适应长度不同的文本
    因此 TTSTokenizer 返回的值必须是 [1, 100] 范围内的整数。

    例子: "xaɪ˧˥" -> tensor([3, 4, 2, 7, 9])
    """

    def __init__(self):
        self.mapping = {
            " ": 1,
            "0": 2,
            "1": 3,
            "2": 4,
            "3": 5,
            "4": 6,
            "5": 7,
            "a": 8,
            "e": 9,
            "f": 10,
            "h": 11,
            "i": 12,
            "k": 13,
            "l": 14,
            "m": 15,
            "n": 16,
            "o": 17,
            "p": 18,
            "r": 19,
            "s": 20,
            "t": 21,
            "u": 22,
            "x": 23,
            "y": 24,
            'ø': 25,
            'ŋ': 26,
            'œ': 27,
            'Ǿ': 28,
            'ɐ': 29,
            'ɔ': 30,
            'ɕ': 31,
            'ə': 32,
            'ɛ': 33,
            'ɤ': 34,
            'ɿ': 35,
            'ʂ': 36,
            'ʅ': 37,
            'ʐ': 38,
            '—': 39,
            '…': 40,
            '。': 41,
            '.': 41,  # '.' 和 '。' 都映射到同一个 ID
            '！': 42,
            '!': 42,  # '!' 和 '！' 都映射到同一个 ID
            '，': 43,
            ',': 43,
            '、': 43,  # '、', '，' 和 ',' 都映射到同一个 ID
            '：': 44,
            ':': 44,  # ':' 和 '：' 都映射到同一个 ID
            '；': 45,
            ';': 45,  # ';' 和 '；' 都映射到同一个 ID
            '？': 46,
            '?': 46  # '?' 和 '？' 都映射到同一个 ID
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
    tokenizer = TTSTokenizer()
    tokens = tokenizer(text)
    print(tokens)  # Should print a tensor of token IDs
    print(tokens.shape)  # Should match the length of the input text