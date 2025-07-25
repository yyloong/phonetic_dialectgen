### 请先添加 bigvgan22HZ 目录路径
import sys
sys.path.append("./bigvgan22HZ")
import bigvgan
import soundfile as sf
import torch

class Load_Bigvgan:
    def __init__(self, model_name="bigvgan22HZ"):
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


if __name__ == "__main__":
    model = Load_Bigvgan()
    ### 获取梅尔频谱
    #mel = model.get_spectrogram("/home/u-longyy/Vocoder/test.wav")
    #mel = mel.to(model.device)
    mel=torch.load('your mel path').reshape(1,80,-1).to(model.device)
    print(mel.shape)
    ### 转化梅尔频谱
    model.spectrogram_to_wave(mel, "your save wav")
