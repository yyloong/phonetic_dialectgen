import torch
from config import GlowTTSConfig
from tokenizer import TTSTokenizer
from inference import load_model_from_checkpoint
from load_bigvgan import Load_Bigvgan

# 导入 text_to_IPA
import os
import sys

sys.path.append(os.path.abspath("/home/u-shigc/phonetic_dialectgen/IPA"))
from text_to_IPA import text_to_IPA


def main():
    # 1. 加载模型

    # 如果检查点包含配置，则可以直接加载

    # 混合模型
    checkpoint_path = "./weights/hybrid.pth"
    # 普通话模型
    # checkpoint_path = "./weights/mandarin.pth"
    # 粤语模型
    # checkpoint_path = "./weights/cantonese.pth"

    # 如果是仅包含模型权重的文件，还需要提供 config
    # checkpoint_path = "./weights/best_model.pth"
    # checkpoint_path = "./outputs/best_model.pth"
    # checkpoint_path = "./restart/best_model.pth"

    config = GlowTTSConfig(
        num_chars=47,
        out_channels=80,
        # 编码器参数
        encoder_type="rel_pos_transformer",
        encoder_params={
            "kernel_size": 3,
            "dropout_p": 0.1,
            "num_layers": 12,  # 从 6 增加到 12
            "num_heads": 8,  # 从 2 增加到 8
            "hidden_channels_ffn": 1024,  # 从 768 增加到 1024
            "input_length": None,
        },
        # 编码器隐藏层 - 这些是分开的参数
        hidden_channels_enc=256,  # 从 192 增加到 256
        hidden_channels_dec=256,  # 从 192 增加到 256
        hidden_channels_dp=400,  # 从 256 增加到 400
        # Flow 参数
        num_flow_blocks_dec=16,  # 从 12 增加到 16
        num_block_layers=6,  # 从 4 增加到 6
    )

    model, config = load_model_from_checkpoint(checkpoint_path, config=config)

    # 2. 准备输入文本
    Chinese_text = "对于死水一潭的中国科幻圈可谓重大利好。"  # 中文文本
    language = "pinyin"  # "pinyin" 或 "jyutping"
    text, failed_words, success = text_to_IPA(Chinese_text, language)
    # text = "tʂɤ51 pu51 tian51 iŋ215 khou215 pei55 xən215 paŋ51 ， pu51 ʐu35 tʂou55 muo51 i55 tɕhi215 tɕhy51 khan51 ？"

    # 3. 文本预处理
    tokenizer = TTSTokenizer()

    # 将文本转换为token序列
    token_ids = tokenizer(text)

    # 转换为tensor
    text_input = torch.LongTensor(token_ids).unsqueeze(0)  # [1, seq_len]
    text_lengths = torch.LongTensor([len(token_ids)])  # [1]

    # 4. 推理
    with torch.no_grad():
        # 生成语音
        outputs = model.inference(x=text_input, x_lengths=text_lengths)

        # 获取梅尔频谱
        mel_spectrogram = outputs["model_outputs"]  # [1, T, C]

    print(f"🎵 生成的梅尔频谱形状: {mel_spectrogram.shape}")

    vocoder = Load_Bigvgan()
    mel_spectrogram = mel_spectrogram.to(vocoder.device).transpose(
        1, 2
    )  # 转置为 [1, C, T]
    out_path = "output.wav"
    vocoder.spectrogram_to_wave(mel_spectrogram, out_path)
    print(f"🎵 音频已保存为 {out_path}")


if __name__ == "__main__":
    main()
