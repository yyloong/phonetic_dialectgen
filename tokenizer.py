import torch
import random
import json

# TODO: TTSTokenizer
# TODO: 在这里填入所有可能出现在 IPA 字符串中的字符总数: 46 这个值记为 n


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
            "\u00f8": 25,
            "\u014b": 26,
            "\u0153": 27,
            "\u01fe": 28,
            "\u0250": 29,
            "\u0254": 30,
            "\u0255": 31,
            "\u0259": 32,
            "\u025b": 33,
            "\u0264": 34,
            "\u027f": 35,
            "\u0282": 36,
            "\u0285": 37,
            "\u0290": 38,
            "\u2014": 39,
            "\u2026": 40,
            "\u3002": 41,
            "\uff01": 42,
            "\uff0c": 43,
            "\uff1a": 44,
            "\uff1b": 45,
            "\uff1f": 46,
        }

    def __call__(self, text):
        # if self.dummy == True:
        #     # 如果是 dummy 模式，随机生成 token_ids
        #     tokens = [random.randint(1, 99) for _ in range(len(text))]
        #     tokens = range(1, 61)
        #     return torch.tensor(tokens, dtype=torch.long)
        # TODO: add real tokenization logic
        # you need to define a mapping from characters to token IDs (映射关系可以随意定义)
        # and return the token IDs as a tensor
        # you need to consider all possible characters in the IPA string
        # 需考虑到空格，标点符号和其他特殊字符
        token_ids = [self.mapping.get(char, 0) for char in text]
        return torch.tensor(token_ids, dtype=torch.long)


if __name__ == "__main__":
    # Example usage
    print("Testing TTSTokenizer...")
    text = "i˨ pən˨˩˦ ʂu˥˥ ， i˧˥ kɤ˧ ku˥˩ ʂɚ˥˩ ， u˧˥ ɕjɛn˥˩ ti˥˩ kɤ ŋ˥ 。"
    print(len(text))
    print(f"Tokenizing {text}...")
    tokenizer = TTSTokenizer()
    print(tokenizer(text))  # Example usage, should print a tensor of token IDs
