import os.path as osp
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset
import pandas as pd
from from_IPA_to_Tensor.IPA_to_Tensor import ipa_to_tensor


class TTSDataset(Dataset):
    def __init__(
        self,
        mandarin_file,
        cantonese_file,
        root_path,
        mandarin_num=49730,
        cantonese_num=18975,
    ):
        mandarin = pd.read_csv(mandarin_file)
        cantonese = pd.read_csv(cantonese_file)
        self.data_df = pd.concat(
            [mandarin.sample(mandarin_num), cantonese.sample(cantonese_num)]
        )
        print(self.data_df.head())
        self.text_tokenizer = ipa_to_tensor
        self.root_path = root_path

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        mel_tensor: 音频的梅尔谱图张量 (Frequency x Time)
        text_tensor: 文本的张量表示
        """
        data = self.data_df.iloc[idx]
        mel_path = str(data["audio"]) + ".pt"  # 存储的是 [B, C, T] 格式的 mel-spec
        mel_tensor = torch.load(osp.join(self.root_path, mel_path), map_location="cpu")

        text_tensor = self.text_tokenizer(data["IPA"])
        return {
            "token_id": text_tensor,
            "token_id_lengths": len(text_tensor),
            "mel": mel_tensor.squeeze(0).transpose(
                0, 1
            ),  # [T, C] 方便后续使用 pad_sequence
            "mel_length": mel_tensor.shape[-1],
        }

    def collate_fn(self, batch):
        """为 GlowTTS 定制的 collate_fn"""
        # 提取数据
        token_id = [item["token_id"] for item in batch]
        token_id_lengths = [item["token_id_lengths"] for item in batch]
        mels = [item["mel"] for item in batch]
        mel_lengths = [item['mel_length'] for item in batch]

        # 文本数据 padding
        token_id_tensor = pad_sequence(
            token_id, batch_first=True, padding_value=self.get_pad_id()
        )

        # 梅尔频谱数据 padding
        mel_tensor = pad_sequence(
            mels, batch_first=True, padding_value=self.get_pad_id()
        )  # [B, T, C]
        mel_tensor = mel_tensor.transpose(1, 2)  # [B, C, T]

        return {
            "token_ids": token_id_tensor,
            "token_ids_lengths": torch.LongTensor(token_id_lengths),
            "mel_input": mel_tensor,  # [B, C, T]
            "mel_lengths": torch.LongTensor(mel_lengths),
        }

    def get_pad_id(self):
        """返回填充 text token 的 ID"""
        return 0  # 或者你的实际 pad_id
