import torch
import argparse
from load_save_checkpoint import load_checkpoint
from from_IPA_to_Tensor.IPA_to_Tensor import ipa_to_tensor
from load_config import Load_config
from load_bigvgan import Load_Bigvgan
from Character_to_IPA.Text_to_IPA import text_to_IPA


def main(args):

    config = Load_config.load_config_toml(args.config_path)
    model = Load_config.load_model(config)
    model, _, _, _, _, _ = load_checkpoint(model, None, None, args.checkpoint_path)

    # 2. 准备输入文本
    text, _, _ = text_to_IPA(args.text, args.language)

    # 3. 文本预处理

    # 将文本转换为token序列
    token_ids = ipa_to_tensor(text)

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
    '''从命令行接受参数'''
    parser = argparse.ArgumentParser(description="Train a model with TOML config")

    parser.add_argument(
        "--config_path",
        default="Tomls_config/Glow_TTS.toml",
        help="Path to TOML config file",
    )

    parser.add_argument(
        "--checkpoint_path",
        default='Glow-TTS_output/checkpoint_step_4999.pth',
        help="Path to your checkpont",
    )
    parser.add_argument(
        "--text",
        default="夕阳把天空染成温柔的橘粉色，晚风带着草木的清香掠过窗台，远处传来几声归鸟的轻啼。路灯次第亮起，晕开一圈圈暖黄的光，给渐暗的街道披上了一层朦胧的纱。此刻无需多言，只需静静感受这份由喧嚣渐入宁静的惬意。",
        help="The text you want to input",
    )
    parser.add_argument(
        "--language", default='jyutping', help="The language you want to output"
    )
    parser.add_argument(
        "--output_path",
        default='output.wav',
        help='The path you want to save the wav fiel',
    )
    args = parser.parse_args()
    main(args)
