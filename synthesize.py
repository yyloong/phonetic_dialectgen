import torch
from model import GlowTTS
from tokenizer import TTSTokenizer
from bigvgan22HZ import Load_Bigvgan
from text_to_IPA import text_to_IPA


def load_model_from_checkpoint(checkpoint_path, config=None):
    """从检查点加载模型进行推理"""
    # 1. 加载检查点
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    # 2. 获取配置（如果没有提供的话）
    if config is None:
        config = checkpoint.get("config", None)
        if config is None:
            raise ValueError("检查点中没有配置信息，请手动提供配置")
    # config.inference_noise_scale = 0.0  # 推理时不使用噪声缩放
    # 3. 创建模型
    model = GlowTTS(config)
    # 4. 加载模型权重
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    # 5. 设置为评估模式
    model.eval()
    print(f"✅ 模型已从检查点加载: {checkpoint_path}")
    print(
        f"📊 检查点信息: 步骤 {checkpoint.get('total_steps_done', 'N/A')}, "
        f"Epoch {checkpoint.get('epochs_done', 'N/A')}, "
        f"损失: {checkpoint.get('best_loss', 'N/A')}"
    )
    return model, config

def synthesize(checkpoint_path, text, language="pinyin"):
    """使用Glow-TTS模型合成语音"""
    # 1. 加载模型
    model, config = load_model_from_checkpoint(checkpoint_path)
    # 2. 准备输入文本
    text, failed_words, success = text_to_IPA(text, language)
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


def main():
    # 混合模型
    checkpoint_path = "./weights/hybrid.pth"
    # 普通话模型
    # checkpoint_path = "./weights/mandarin.pth"
    # 粤语模型
    # checkpoint_path = "./weights/cantonese.pth"

    text = "做人如果冇梦想，咁同條咸鱼有咩分别？"  # 输入文本
    language = 'jyutping'  # pinyin, jyutping  

    synthesize(checkpoint_path, text, language)
    print("✅ 合成完成！")


if __name__ == "__main__":
    main()
