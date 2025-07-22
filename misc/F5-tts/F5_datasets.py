from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
from create_IPA import convert_sentence_to_codes
from torch.nn.utils.rnn import pad_sequence


class VariableLengthMelDataset(Dataset):
    def __init__(
        self,
        directories,
        paths,
        file_extension=".pt",
    ):
        self.ipa = []
        self.directories = (
            directories if isinstance(directories, list) else [directories]
        )
        self.paths = paths if isinstance(paths, list) else [paths]
        self.pt_files = []
        for directory in self.directories:
            self.load_mel(directory,file_extension)
        for path in self.paths:
            self.load_IPA(path)
        if len(self.pt_files) != len(self.ipa):
            raise ValueError(
                f"Number of mel-spectrograms ({len(self.pt_files)}) does not match number of IPA sequences ({len(self.ipa)})"
            )

    def load_mel(self, directory, file_extension):
        '''self.pt_files += [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(file_extension)
        ]'''
        self.pt_files +=[
            os.path.join(directory,f"{i}"+file_extension)
            for i in range(1,len(os.listdir(directory))+1)
        ]
        if not self.pt_files:
            raise ValueError(
                f"No {file_extension} files found in directory {directory}"
            )

    def load_IPA(self, path):
        df = pd.read_csv(path)
        ipa_list = df['IPA']
        for i in ipa_list:
            ipa = list(convert_sentence_to_codes(i))
            self.ipa.append(torch.tensor(ipa, dtype=torch.long))

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        mel = torch.load(self.pt_files[idx])
        ipa = self.ipa[idx]
        mel_len = mel.shape[-1]
        return [mel.transpose(0, 1), mel_len, ipa]

    @staticmethod
    def collate_fn(batch):
        mel = [sample[0] for sample in batch]
        mel_len = [sample[1] for sample in batch]
        ipa = [sample[2] for sample in batch]
        mel = pad_sequence(mel, batch_first=True)
        ipa = pad_sequence(ipa, batch_first=True)
        mel_len = torch.tensor(mel_len)
        return (mel, mel_len, ipa)


if __name__ == "__main__":
    path = ["/mnt/nas/shared/datasets/voices/AItts/AItts2/data_with_pinyinIPA.csv","/mnt/nas/shared/datasets/voices/AItts/AItts3/data_with_pinyinIPA.csv"]
    dirctory = ["../AItts2mel/melspec","../melspec"]
    data = VariableLengthMelDataset(directories=dirctory, paths=path)
