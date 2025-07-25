from wav_to_melspec import mel_spectrogram
import tqdm
import sys
import multiprocessing as mp
import os
import torch
import librosa


def get_spectrogram(path, turn):
    '''TODO:change the parameter to fit the model if you use another moder'''
    signal, _ = librosa.load(path, sr=22050, mono=True)
    device = f"cuda:{turn%4}"
    signal = torch.Tensor(signal).to(device)
    spectrogram, _ = mel_spectrogram(
        y=signal.squeeze(),
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=None,
    )
    return spectrogram


def parallel_process(wav_dir, save_dir, process_fun, num_workers=mp.cpu_count()):
    '''.wav file to .pt file,wav_dir is the dir of input files the output will be save in save_dir'''
    os.makedirs(save_dir, exist_ok=True)
    listdir = os.listdir(wav_dir)
    wav_paths = [os.path.join(wav_dir, f) for f in listdir if f.endswith(".wav")]
    save_paths = [
        os.path.join(save_dir, f.replace(".wav", ".pt"))
        for f in listdir
        if f.endswith(".wav")
    ]

    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    process_fun,
                    zip(wav_paths, save_paths, list(range(len(wav_paths)))),
                ),
                total=len(wav_paths),
                desc="Processing WAV files",
                file=sys.stdout,
            )
        )

    return results


def process_file(wav_path, save_path, turn):
    if os.path.exists(save_path):
        return save_path
    try:
        mel_spec = get_spectrogram(wav_path, turn)
        torch.save(mel_spec, save_path)
    except:
        print(f"{os.path.basename(save_path)}", end=",")
    return save_path


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    wave_path = "your path"
    save_path = "your path"
    parallel_process(wave_path, save_path, process_file)
