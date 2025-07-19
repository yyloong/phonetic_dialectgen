import torch
from model import GlowTTS
from config import GlowTTSConfig
from tokenizer import ShuTokenizer
import sys
import bigvgan as bigvgan
import soundfile as sf
import yaml

sys.path.append("/home/u-wuhc/backup/bigvgan22HZ")
with open('shupin.yaml', 'r', encoding='utf-8') as file:
    mapping = yaml.safe_load(file)

def convert_text(text):
    converted_text = []
    for char in text:
        if char in mapping:
            converted_text.append(mapping[char])
        else:
            converted_text.append(char)
    return ' '.join(converted_text)

class Load_Bigvgan:
    def __init__(self, model_name="/home/u-wuhc/backup/bigvgan22HZ"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = bigvgan.BigVGAN.from_pretrained(model_name)
        self.model.remove_weight_norm()
        self.model = self.model.eval().to(self.device)
        self.h = self.model.h

    def spectrogram_to_wave(self, spectrogram, path):
        with torch.inference_mode():
            wavgen = self.model(spectrogram)
        wav_gen_float = wavgen.squeeze(0).cpu()
        sf.write(path, wav_gen_float[0].numpy(), self.model.h.sampling_rate)

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


def main():
    # 如果检查点包含配置，则可以直接加载
    checkpoint_path = "./outputs/checkpoint_step_164999.pth" 

    # 如果是仅包含模型权重的文件，还需要提供 config
    # checkpoint_path = "./outputs/best_model.pth"  

    config = GlowTTSConfig(
        num_chars=39,
        out_channels=80,
        encoder_type="rel_pos_transformer",
        encoder_params={
            "kernel_size": 3,
            "dropout_p": 0.1,
            "num_layers": 12,
            "num_heads": 8,
            "hidden_channels_ffn": 1024,
            "input_length": None,
        },
        hidden_channels_enc=256,
        hidden_channels_dec=256,
        hidden_channels_dp=400,
        num_flow_blocks_dec=16,
        num_block_layers=6,
    )

    model, config = load_model_from_checkpoint(checkpoint_path, config=config)
    
    text = "你好，我是一个语音合成模型。"
    text = convert_text(text)
    print(f"转换后的文本: {text}")
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

    vocoder = Load_Bigvgan()
    mel_spectrogram = mel_spectrogram.to(vocoder.device).transpose(1, 2)  # 转置为 [1, C, T]
    out_path = "output.wav"
    vocoder.spectrogram_to_wave(mel_spectrogram, out_path)
    print(f"🎵 音频已保存为 {out_path}")

if __name__ == "__main__":
    main()