import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd

class TTSDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        csv_file,
        root_path
    ):
        self.data_df = pd.read_csv(csv_file)
        print(self.data_df.head())
        self.text_tokenizer = tokenizer
        self.root_path = root_path

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        mel_tensor: 音频的梅尔谱图张量 (Frequency x Time)
        text_tensor: 文本的张量表示
        """
        data = self.data_df.iloc[idx]
        mel_path = str(data["audio"]) + ".pt"      # 存储的是 [B, C, T] 格式的 mel-spec
        mel_tensor = torch.load(osp.join(self.root_path, mel_path), map_location='cpu') 
        text_tensor = self.text_tokenizer(data["IPA"])
        return {
            "token_id": text_tensor,
            "token_id_lengths": len(text_tensor),
            "mel": mel_tensor      # [B, C, T]
        }
    
    def collate_fn(self, batch):
        """为 GlowTTS 定制的 collate_fn"""
        # 提取数据
        token_id = [item["token_id"] for item in batch]
        token_id_lengths = [item["token_id_lengths"] for item in batch]
        mels = [item["mel"] for item in batch]
        
        # 计算梅尔频谱长度
        mel_lengths = [mel.shape[1] for mel in mels]  # 梅尔频谱的时间维度
        
        # 文本数据 padding
        max_text_len = max(token_id_lengths)
        token_id_tensor = torch.LongTensor(len(batch), max_text_len).fill_(self.get_pad_id())
        for i, length in enumerate(token_id_lengths):
            token_id_tensor[i, :length] = torch.LongTensor(token_id[i])
        
        # 梅尔频谱数据 padding
        max_mel_len = max(mel_lengths)
        mel_dim = mels[0].shape[0]  # 频率维度
        mel_tensor = torch.zeros(len(batch), mel_dim, max_mel_len)
        for i, mel in enumerate(mels):
            mel_length = mel.shape[1]
            mel_tensor[i, :, :mel_length] = mel
        
        # 返回 GlowTTS 期望的格式
        return {
            "token_ids": token_id_tensor,                    
            "token_ids_lengths": torch.LongTensor(token_id_lengths),  
            "mel_input": mel_tensor,   # [B, C, T]
            "mel_lengths": torch.LongTensor(mel_lengths)    
        }

    def get_pad_id(self):
        """返回填充 text token 的 ID"""
        return 0  # 或者你的实际 pad_id