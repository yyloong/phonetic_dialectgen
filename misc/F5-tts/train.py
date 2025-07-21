# training script.
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import torch
from F5_datasets import VariableLengthMelDataset
from Dit import DiT
from cfm import CFM
from Trainer import Trainer
from torch.utils.data import random_split


def main():
    model_cls = DiT(
        dim=1024,
        depth=16,#22,
        heads=16,
        ff_mult=2,
        text_dim=480,#512
        text_mask_padding=True,
        qk_norm=None,  # null | rms_norm
        conv_layers=4,
        pe_attn_head=None,
        attn_backend="torch",  # torch | flash_attn
        attn_mask_enabled=False,
        checkpoint_activations=False,  # recompute activations and save memory for extra compute)
        text_num_embeds=46,
        mel_dim=80,
    )
    # set model

    model = CFM(
        transformer=model_cls,
    )
    
    #cpt=torch.load("")
    #model.load_state_dict(cpt['model_state_dict'])

    model = DDP(
        model.cuda(local_rank),
        device_ids=[local_rank],
        find_unused_parameters=True,
    ).module

    # init trainer
    trainer = Trainer(
        model,
        epochs=200,
        learning_rate=7.5e-5,
        num_warmup_updates=20,
        batch_size_per_gpu=8,
        max_samples=64,
        local=local_rank,
    )
    path = ["/mnt/nas/shared/datasets/voices/AItts/AItts2/data_with_pinyinIPA.csv","/mnt/nas/shared/datasets/voices/AItts/AItts3/data_with_pinyinIPA.csv"]
    dirctory = ["../AItts2mel/melspec","../melspec"]
    train_dataset = VariableLengthMelDataset(
        directories=dirctory, paths=path 
    )
    #train_dataset,_=random_split(train_dataset,[16,len(train_dataset)-16])
    trainer.train(
        train_dataset,
    )


if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    main()