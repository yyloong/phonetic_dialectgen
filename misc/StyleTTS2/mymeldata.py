import os.path as osp
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
# from torch import nn
# import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i


class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes


np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {"n_fft": 2048, "win_length": 1200, "hop_length": 300}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
mean, std = -4, 4


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        root_path,
        sr=24000,
        data_augmentation=False,
        validation=False,
        min_length=50,  
    ):
        # 简化数据格式处理 - 只需要路径和文本
        self.data_df = pd.read_csv(csv_file)
        print(self.data_df.head())
        self.text_cleaner = TextCleaner()
        self.sr = sr
        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        self.root_path = root_path

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        data = self.data_df.iloc[idx]
        path = str(data["audio"]) + ".wav"

        wave, text_tensor = self._load_tensor(data)  
        mel_tensor = preprocess(wave).squeeze()
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[
            :, : (length_feature - length_feature % 2)
        ]

        return acoustic_feature, text_tensor, path, wave

    def _load_tensor(self, data):
        wave_path = str(data["audio"]) + ".wav"
        text = data["text"]
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            print(wave_path, sr)

        wave = np.concatenate(
            [np.zeros([5000]), wave, np.zeros([5000])], axis=0
        )

        text = self.text_cleaner(text)
        text.insert(0, 0)
        text.append(0)
        text = torch.LongTensor(text)

        return wave, text


class Collater(object):
    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        lengths = [b[0].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][0].size(0)
        max_mel_length = max([b[0].shape[1] for b in batch])
        max_text_length = max([b[1].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        paths = ["" for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]

        for bid, (mel, text, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            waves[bid] = wave

        return waves, texts, input_lengths, mels, output_lengths


def build_dataloader(
    path_list,
    root_path,
    validation=False,
    min_length=50,
    batch_size=4,
    num_workers=1,
    device="cpu",
    collate_config={},
    dataset_config={},
):
    dataset = FilePathDataset(
        path_list,
        root_path,
        min_length=min_length,
        validation=validation,
        **dataset_config,
    )
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    return data_loader


def save_preprocessed_data(dataloader):
    processed_data = []
    for i, batch in enumerate(dataloader):
        waves, texts, input_lengths, mels, output_lengths = batch
        sample = {
            "wave": waves[0],
            "text": texts[0],
            "input_length": input_lengths[0],
            "mel": mels[0],
            "output_length": output_lengths[0],
        }
        processed_data.append(sample)
        if i % 100 == 0:
            print(f"Processed {i} samples")

    torch.save(processed_data, "preprocessed_train_data.pt")
    print(f"Saved {len(processed_data)} samples")


# 加载预处理的数据
def load_preprocessed_data(file_path):
    return torch.load(file_path, weights_only=False)


if __name__ == "__main__":
    # Example usage
    csv_file = "Data/aitts1/data.csv"  # Replace with your actual CSV file path
    root_path = "Data/aitts1/data"  # Replace with your actual root path
    dataloader = build_dataloader(
        csv_file,
        root_path,
        validation=False,
        batch_size=4,
        num_workers=2,
    )
    save_preprocessed_data(dataloader)
    data = load_preprocessed_data("preprocessed_train_data.pt")
    print(data[0])  # Print the first preprocessed sample