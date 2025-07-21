from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
from create_IPA import convert_sentence_to_codes


class VariableLengthMelDataset(Dataset):
    def __init__(
        self,
        directories,
        paths,
        cache=False,
        add_max_len=2000,
        file_extension=".pt",
    ):
        self.mel = []
        self.ipa = []
        self.mel_max_len = 0
        self.ipa_max_len = 0
        self.directories = directories if isinstance(directories, list) else [directories]
        self.paths = paths if isinstance(paths, list) else [paths]
        self.cache=cache
        self.pt_files=[]
        if cache==False:
            self.mel_max_len=add_max_len

        for directory in self.directories:
            self.load_mel(directory, cache,file_extension)
        for path in self.paths:
            self.load_IPA(path)
        if len(self.pt_files) != len(self.ipa):
            raise ValueError(f"Number of mel-spectrograms ({len(self.pt_files)}) does not match number of IPA sequences ({len(self.ipa)})")

        
    def load_mel(self, directory, cache,file_extension):
        self.pt_files +=[
            os.path.join(directory,f"{i}"+file_extension)
            for i in range(1,len(os.listdir(directory))+1)
        ]
        if not self.pt_files:
            raise ValueError(
                f"No {file_extension} files found in directory {directory}"
            )
        if cache==True:
        # 预加载所有数据
            for f in self.pt_files:
                mel = torch.load(f)
                if len(mel.shape) == 3:
                    if mel.shape[0] != 1:
                        raise ValueError(
                            f"Expected batch_size=1 in {f}, got shape {mel.shape}"
                        )
                    mel = mel[0]
                elif len(mel.shape) != 2:
                    raise ValueError(
                        f"Expected 2D or 3D tensor in {f}, got shape {mel.shape}"
                    )
                self.mel_max_len = (
                    mel.shape[-1] if self.mel_max_len < mel.shape[-1] else self.mel_max_len
                )
                self.mel.append(mel)

    def load_IPA(self, path):
        df = pd.read_csv(path)
        ipa_list = df['IPA']
        for i in ipa_list:
            ipa = list(convert_sentence_to_codes(i))
            self.ipa.append(torch.tensor(ipa,dtype=torch.long))
            self.ipa_max_len = (
                len(ipa) if self.ipa_max_len < len(ipa) else self.ipa_max_len
            )

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        if self.cache==True:
            mel = self.mel[idx]
        else:
            mel = torch.load(self.pt_files[idx])
        ipa = self.ipa[idx]
        mel_len = mel.shape[-1]
        ipa_len = ipa.shape[-1]
        padded_mel = torch.zeros(mel.shape[-2], self.mel_max_len)
        padded_mel[:, :mel_len] = mel
        padded_ipa = torch.zeros(self.ipa_max_len,dtype=torch.long)
        padded_ipa[:ipa_len] = ipa
        return padded_mel, mel_len, padded_ipa, ipa_len


if __name__ == "__main__":
    path = "/mnt/nas/shared/datasets/voices/AItts/AItts3/data_with_pinyinIPA.csv"
    dirctory = "../melspec"
    data = VariableLengthMelDataset(
        directories=dirctory, paths=path
    )
    print(data.ipa_max_len)
