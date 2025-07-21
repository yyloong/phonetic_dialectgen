from __future__ import annotations

from torch.utils.data.distributed import DistributedSampler
import gc
import math
import os

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from cfm import CFM
#from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from utils import default, exists


# trainer
def save_checkpoint(model, optimizer,schedule, epoch, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  # 保存原始模型的 state_dict
        'optimizer_state_dict': optimizer.state_dict(),
        'schedule_state_dict' : schedule.state_dict()
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")



class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs: int,
        learning_rate: float,
        num_warmup_updates: int = 200,
        batch_size_per_gpu: int = 32,
        max_samples: int = 32,
        local=None
    ):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_warmup_updates = num_warmup_updates
        self.batch_size_per_gpu = batch_size_per_gpu
        self.max_samples = max_samples
        self.device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.local=local

    def train(self, train_dataset):
        self.model.train()
        sample= DistributedSampler(train_dataset,rank=self.local,seed=0,shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_gpu,
            sampler=sample,
            collate_fn=train_dataset.collate_fn
        )
        total_updates = math.ceil(len(train_dataloader)) * self.epochs
        warmup_updates = self.num_warmup_updates
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )

        for epoch in range(self.epochs):
            for mel,mel_lengths,text in train_dataloader:
                mel_spec = mel.to(self.device)
                text_inputs = text.to(self.device)
                mel_lengths = mel_lengths.to(self.device)
                loss, _, _ = self.model(mel_spec, text=text_inputs, lens=mel_lengths)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
            if self.local== 0:
                save_checkpoint(self.model,self.optimizer,self.scheduler,epoch,f"F5-tts/new_checkpoint/{(epoch+1)%20}_checkpoint.pt")
