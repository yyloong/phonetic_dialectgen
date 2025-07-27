from wav_to_melspec import mel_spectrogram
from tqdm import tqdm
import sys
import multiprocessing as mp
import os
import torch
from wav_to_melspec import get_spectrogram



def parallel_process(wav_dir, save_dir, process_fun, num_workers=int(mp.cpu_count()/2)):
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
                    zip(wav_paths, save_paths),
                ),
                total=len(wav_paths),
                desc="Processing WAV files",
                file=sys.stdout,
            )
        )

    return results


def process_file(wav_path, save_path):
    if os.path.exists(save_path):
        return save_path
    try:
        mel_spec = get_spectrogram(wav_path)
        torch.save(mel_spec, save_path)
    except:
        print(f"{os.path.basename(save_path)}", end=",")
    return save_path


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    wave_path = "your path"
    save_path = "your path"
    parallel_process(wave_path, save_path, process_file)
