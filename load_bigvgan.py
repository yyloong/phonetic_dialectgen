### 请先添加 bigvgan22HZ 目录路径
import sys
sys.path.append("/mnt/nas/shared/datasets/voices/bigvgan22HZ")
import bigvgan as bigvgan
import soundfile as sf
import librosa
import torch
# from meldataset import get_mel_spectrogram


class Load_Bigvgan:
    def __init__(self, model_name="/mnt/nas/shared/datasets/voices/bigvgan22HZ"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = bigvgan.BigVGAN.from_pretrained(model_name)
        self.model.remove_weight_norm()
        self.model = self.model.eval().to(self.device)
        self.h = self.model.h

    # def get_spectrogram(self, path,h=None):
    #     if h==None:
    #         h=self.h.sampling_rate
    #     wav, sr = librosa.load(path, sr=h, mono=True)
    #     wav = torch.FloatTensor(wav).unsqueeze(0)
    #     mel = get_mel_spectrogram(wav, self.h)
    #     return mel

    def spectrogram_to_wave(self, spectrogram, path):
        with torch.inference_mode():
            wavgen = self.model(spectrogram)
        wav_gen_float = wavgen.squeeze(0).cpu()
        sf.write(path, wav_gen_float[0].numpy(), self.model.h.sampling_rate)


if __name__ == "__main__":
    model = Load_Bigvgan()
    
    # mel = torch.load("data/48.pt")
    # mel = mel.unsqueeze(0)  # 添加 batch 维度
    # mel = mel.to(model.device)

    mel = torch.load("mel_output.pth")
    mel = mel.to(model.device).transpose(1, 2)
     
    ### 转化梅尔频谱
    model.spectrogram_to_wave(mel, "output.wav")
    print("🎵 音频已保存为 output.wav")
