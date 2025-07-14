import torch
import random

# TODO: TTSTokenizer
# TODO: 在这里填入所有可能出现在 IPA 字符串中的字符总数: _____ (例如 100)， 这个值记为 n


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

    def __init__(self, dummy=True):
        self.dummy = dummy
        # TODO: add code if necessary

    def __call__(self, text):
        if self.dummy == True:
            # 如果是 dummy 模式，随机生成 token_ids
            tokens = [random.randint(1, 99) for _ in range(len(text))]
            return torch.tensor(tokens, dtype=torch.long)
        # TODO: add real tokenization logic
        # you need to define a mapping from characters to token IDs (映射关系可以随意定义)
        # and return the token IDs as a tensor
        # you need to consider all possible characters in the IPA string
        # 需考虑到空格，标点符号和其他特殊字符


if __name__ == "__main__":
    # Example usage
    print("Testing TTSTokenizer...")
    text = "xaɪ˧˥"
    print(f"Tokenizing {text}...")
    tokenizer = TTSTokenizer()
    print(
        tokenizer("xaɪ˧˥")
    )  # Example usage, should print a tensor of token IDs
