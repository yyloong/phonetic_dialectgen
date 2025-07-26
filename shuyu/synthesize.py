import torch
from shuyu.model import GlowTTS
from shuyu.tokenizer import ShuTokenizer
from bigvgan22HZ import Load_Bigvgan
import yaml

with open('shuyu/shupin.yaml', 'r', encoding='utf-8') as file:
    mapping = yaml.safe_load(file)

def convert_text(text):
    converted_text = []
    for char in text:
        if char in mapping:
            converted_text.append(mapping[char])
        else:
            converted_text.append(char)
    return ' '.join(converted_text)

def load_model_from_checkpoint(checkpoint_path, config=None):
    """从检查点加载模型进行推理"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if config is None:
        config = checkpoint.get('config', None)
        if config is None:
            raise ValueError("检查点中没有配置信息，请手动提供配置")
    # config.inference_noise_scale = 0.0  # 推理时不使用噪声缩放
    model = GlowTTS(config)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ 模型已从检查点加载: {checkpoint_path}")
    print(f"📊 检查点信息: 步骤 {checkpoint.get('total_steps_done', 'N/A')}, "
          f"Epoch {checkpoint.get('epochs_done', 'N/A')}, "
          f"损失: {checkpoint.get('best_loss', 'N/A')}")
    return model, config

def synthesize_sichuan(checkpoint_path, text):
    model, config = load_model_from_checkpoint(checkpoint_path)
    
    text = ' ' + text
    text = convert_text(text)
    tokenizer = ShuTokenizer()
    token_ids = tokenizer(text)
    
    # 转换为tensor
    text_input = torch.LongTensor(token_ids).unsqueeze(0)  # [1, seq_len]
    text_lengths = torch.LongTensor([len(token_ids)])      # [1]
    
    with torch.no_grad():
        outputs = model.inference(
            x=text_input,
            x_lengths=text_lengths
        )
        mel_spectrogram = outputs["model_outputs"]  # [1, T, C]
        
    print(f"🎵 生成的梅尔频谱形状: {mel_spectrogram.shape}")

    vocoder = Load_Bigvgan('bigvgan22HZ/model')
    mel_spectrogram = mel_spectrogram.to(vocoder.device).transpose(1, 2)  # 转置为 [1, C, T]
    out_path = "output.wav"
    vocoder.spectrogram_to_wave(mel_spectrogram, out_path)
    print(f"🎵 音频已保存为 {out_path}")



def main():
    checkpoint_path = "./shuyu/weights/sichuan.pth"
    text = "你好，欢迎使用四川话语音合成系统！我的名字叫做小川。"
    synthesize_sichuan(checkpoint_path, text)

if __name__ == "__main__":
    # 需要按 python 包的方式运行
    # python -m shuyu.synthesize
    main()