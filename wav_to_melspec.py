from tqdm import tqdm
import pandas as pd
import subprocess
from torch.nn.utils.rnn import pad_sequence
import multiprocessing as mp
import os
import torch
from torch.utils.data import Dataset
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import librosa
import sys


class MelDtaset(Dataset):
    def __init__(self, mel_dir):
        self.mel_paths = sorted(
            [os.path.join(mel_dir, f) for f in os.listdir(mel_dir) if f.endswith(".pt")]
        )

    def __len__(self):
        return len(self.mel_paths)

    def __getitem__(self, index):
        mel = torch.load(self.mel_paths[index])
        length = mel.shape[-1]
        return {
            "mel": mel,
            "length": length,
        }

    @staticmethod
    def collate_mel_batch(batch):
        """
        batch: List[{"mel": Tensor [n_mels, T], "length": int}]
        return: padded mel [B, n_mels, T_max], lengths [B]
        """
        mels = [item["mel"].transpose(0, 1) for item in batch]  # 转为 [T, n_mels]
        lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

        # pad to [B, T_max, n_mels]
        padded = pad_sequence(mels, batch_first=True)

        # 转回 [B, n_mels, T_max]
        padded = padded.transpose(1, 2)

        return padded, lengths

    def loader(self, batch_size, shuffle, num_workers):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_mel_batch,
        )


def sp_get_spectrogram(path, turn):
    signal, _ = librosa.load(path, sr=22050, mono=True)
    device = f"cuda:{turn%4}"
    signal = torch.Tensor(signal).to(device)
    spectrogram, _ = mel_spectogram(
        audio=signal.squeeze(),
        sample_rate=22050,
        hop_length=256,
        win_length=1024,  # None,
        n_mels=80,  # 80,
        n_fft=1024,  # 1024,
        f_min=0,  # 0.0,
        f_max=None,  # 8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True,
    )
    return spectrogram


class wav_to_melspec:
    @staticmethod
    def mp3_to_wave(mp3_dir,wav_dir,turn=None):
        os.makedirs(wav_dir, exist_ok=True)
        listdir = os.listdir(mp3_dir)
        mp3_paths = [os.path.join(mp3_dir, f) for f in listdir if f.endswith(".mp3")]
        print(f"len:{len(mp3_paths)}")
        wav_paths = [
            os.path.join(wav_dir, f.replace(".mp3", ".wav"))
            for f in listdir
            if f.endswith(".mp3")
        ]
        for i in range(len(mp3_paths)):
            command = ["ffmpeg", "-i", mp3_paths[i], wav_paths[i]]
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"转换失败: {mp3_paths[i]}, 错误: {e}")

    @staticmethod
    def process_dir(wav_dir, save_dir, turn):
        """
        Process all .wav files in the directory and save their mel spectrograms.
        """
        os.makedirs(save_dir, exist_ok=True)
        listdir = os.listdir(wav_dir)
        wav_paths = [os.path.join(wav_dir, f) for f in listdir if f.endswith(".wav")]
        save_paths = [
            os.path.join(save_dir, f.replace(".wav", ".pt"))
            for f in listdir
            if f.endswith(".wav")
        ]
        for i in range(len(wav_paths)):
            mel_spec = sp_get_spectrogram(wav_paths[i], turn)
            torch.save(mel_spec, save_paths[i])

    @staticmethod
    def process_file(wav_path, save_path, turn):
        mel_spec = sp_get_spectrogram(wav_path, turn)
        torch.save(mel_spec, save_path)
        return save_path  # 否则tqdm无法显示进度

    @staticmethod
    def parallel_process(
        file_or_dir, wav_dir, save_dir, process_fun, num_workers=mp.cpu_count()
    ):
        if file_or_dir == "file":
            os.makedirs(save_dir, exist_ok=True)
            listdir = os.listdir(wav_dir)
            wav_paths = [
                os.path.join(wav_dir, f) for f in listdir if f.endswith(".wav")
            ]
            save_paths = [
                os.path.join(save_dir, f.replace(".wav", ".pt"))
                for f in listdir
                if f.endswith(".wav")
            ]
        elif file_or_dir == "dir":
            wav_paths = wav_dir
            save_paths = save_dir
        else:
            assert False, "file_or_dir must be 'file' or 'dir'"
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


def AItts_mel():
    wave_path_list = [
        f"/mnt/nas/shared/datasets/voices/AItts/AItts{i}/data" for i in range(1, 6)
    ]
    save_path = [
        f"/mnt/nas/shared/datasets/voices/AItts/AItts{i}/melspec" for i in range(1, 6)
    ]
    for i in range(len(wave_path_list)):
        wav_to_melspec.parallel_process(
            "file", wave_path_list[i], save_path[i], wav_to_melspec.process_file
        )


def MDCC():
    wave_path = "/mnt/nas/shared/datasets/voices/MDCC/audio"
    save_path = "/mnt/nas/shared/datasets/voices/MDCC/melspec"
    wav_to_melspec.parallel_process(
        "file", wave_path, save_path, wav_to_melspec.process_file
    )


def word_shk_cantonese():
    wave_path = (
        "/mnt/nas/shared/datasets/voices/wordshk_cantonese_speech/data/wav_files"
    )
    save_path = "/mnt/nas/shared/datasets/voices/wordshk_cantonese_speech/data/melspec"
    wav_to_melspec.parallel_process(
        "file", wave_path, save_path, wav_to_melspec.process_file
    )


def zhvoice_mel():
    root_dir = "/mnt/nas/shared/datasets/voices/zhvoice"
    dir_name = os.listdir("/mnt/nas/shared/datasets/voices/zhvoice/sample")
    mp3_dir_list = [os.path.join(root_dir, dir_name[i]) for i in range(len(dir_name))]
    wav_dir_list = [dir + "wav" for dir in mp3_dir_list]
    save_dir_list = [dir + "mel" for dir in mp3_dir_list]
    for i in wav_dir_list:
        os.makedirs(i, exist_ok=True)
    for i in save_dir_list:
        os.makedirs(i, exist_ok=True)
    mp3_dir_total_list = []
    wav_dir_total_list = []
    save_dir_total_list = []
    for i in range(len(mp3_dir_list)):
        listdir = os.listdir(mp3_dir_list[i])
        print(f"listdir: {len(listdir)}")
        for dir in listdir:
            mp3_concatdir = os.path.join(mp3_dir_list[i], dir)
            if not os.path.isdir(mp3_concatdir):
                continue
            mp3_dir_total_list.append(mp3_concatdir)
            wav_concatdir = os.path.join(wav_dir_list[i], dir)
            os.makedirs(wav_concatdir, exist_ok=True)
            wav_dir_total_list.append(wav_concatdir)
            save_concatdir = os.path.join(save_dir_list[i], dir)
            os.makedirs(save_concatdir, exist_ok=True)
            save_dir_total_list.append(save_concatdir)
    print("mp3_dir_total_list:", len(mp3_dir_total_list))
    print("wav_dir_total_list:", len(wav_dir_total_list))
    wav_to_melspec.parallel_process(
        "dir", mp3_dir_total_list, wav_dir_total_list, wav_to_melspec.mp3_to_wave
    )
    wav_to_melspec.parallel_process(
        "dir", wav_dir_total_list, save_dir_total_list, wav_to_melspec.process_dir
    )


def LibriSpeech_mel():
    root_dir = "/mnt/nas/shared/datasets/voices/LibriSpeech/train-clean-360"

    # 收集所有 speaker/chapter 子目录路径
    chapter_paths = []
    for speaker_id in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
        for chapter_id in os.listdir(speaker_path):
            if chapter_id[-4:] != "_wav":
                continue
            chapter_path = os.path.join(speaker_path, chapter_id)
            if os.path.isdir(chapter_path):
                chapter_paths.append(chapter_path)

    print(chapter_paths)
    # 使用 tqdm 跟踪整体处理进度
    for chapter_path in tqdm(chapter_paths, desc="Processing Chapters"):
        save_path = chapter_path + "_melspec"
        wav_to_melspec.parallel_process(
            "file", chapter_path, save_path, wav_to_melspec.process_file
        )


def CVCorpus_mel():
    paths = [
        # "/mnt/nas/shared/datasets/voices/cv-corpus-22.0-2025-06-20/nan-tw/clips_wav",
        # "/mnt/nas/shared/datasets/voices/cv-corpus-22.0-2025-06-20/yue/clips_wav",
        "/mnt/nas/shared/datasets/voices/cv-corpus-22.0-2025-06-20/zh-HK/clips_wav",
        "/mnt/nas/shared/datasets/voices/cv-corpus-22.0-2025-06-20/zh-TW/clips_wav",
    ]
    for path in paths:
        save_path = path + "_melspec"
        wav_to_melspec.parallel_process(
            "file", path, save_path, wav_to_melspec.process_file
        )


def KeSpeech_mel():
    root_dir = "/mnt/nas/shared/datasets/voices/KeSpeech/Audio"
    save_dir = root_dir
    phase1_mandarin = root_dir.replace("Audio","/Metadata/phase1_mandarin.csv")
    df=pd.read_csv(phase1_mandarin)
    relative_paths= list(df["audio"])
    wav_paths= []
    save_paths= []
    for re_path in relative_paths:
        wav_paths.append(os.path.join(root_dir, re_path))
        save_paths.append(os.path.join(save_dir, re_path.replace(".wav", ".pt").replace("phase","phase_mel")))
        os.makedirs(os.path.dirname(save_paths[-1]), exist_ok=True)
    wav_to_melspec.parallel_process(
        "dir", wav_paths, save_paths, wav_to_melspec.process_file
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    zhvoice_mel()
