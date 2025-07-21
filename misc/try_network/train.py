import torch
import os
from load_checkpoint import load_check
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,random_split
from val_datasets import VariableLengthMelDataset
import commons
from model import SynthesizerTrn, Decoder
from losses import kl_loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def save_checkpoint(model, optimizer,schedule, epoch, filepath, rank):
    if rank == 0:  # 仅在主进程 (rank 0) 保存模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),  # 保存原始模型的 state_dict
            'optimizer_state_dict': optimizer.state_dict(),
            'schedule_state_dict' : schedule.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")


def train(data_loader, local_rank, epochs=100):
    # 模型

    #checkpoint = torch.load('checkpoint/180_checkpoint.pt')
    model=SynthesizerTrn(
            n_vocab=47,
            spec_channels=80,
            inter_channels=512,
            hidden_channels=256,#192,
            filter_channels=792,
            n_heads=4,#2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            use_sdp=False,
        )
    #model.load_state_dict(checkpoint['model_state_dict'])
    net_g = DDP(
        model.cuda(local_rank),
        device_ids=[local_rank],
        find_unused_parameters=True,
    )
    # 优化器
    optim_g = optim.AdamW(net_g.parameters(), lr=1e-4, betas=[0.9, 0.99], eps=1e-9)
    scheduler_g = torch.optim.lr_scheduler.OneCycleLR(
        optim_g, max_lr=1e-3,total_steps=400 
    )
    net_g.train()
    for epoch in range(epochs):
        for mel, mel_len, ipa, ipa_len in data_loader:
            # 生成器前向
            mel = mel.cuda(local_rank)
            mel_len = mel_len.cuda(local_rank)
            ipa = ipa.cuda(local_rank)
            ipa_len = ipa_len.cuda(local_rank)
            (
                gen_mel,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = net_g(ipa, ipa_len, mel, mel_len)
            mel_slice = commons.slice_segments(mel, ids_slice, 64)
            loss_dur = torch.sum(l_length.float())*5
            loss_mel = F.l1_loss(mel_slice, gen_mel) * 45
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)*10
            loss_gen_all = loss_mel + loss_dur + loss_kl#+loss_infer
            if local_rank == 0:
                if epoch % 1 == 0:
                    print("epoch:", epoch)
                    print(
                        f"loss_gen_all:{loss_gen_all.item()}, loss_mel:{loss_mel.item()}, loss_dur:{loss_dur.item()}, loss_kl:{loss_kl.item()}"
                    )
                    print("gmel shape:", gen_mel.shape)
            optim_g.zero_grad()
            loss_gen_all.backward()
            optim_g.step()
        scheduler_g.step()
        if local_rank ==0 and epoch %10==0:
            save_checkpoint(net_g.module,optim_g,scheduler_g,epochs,f"checkpoint/{epoch}_checkpoint.pt",local_rank)
    net_g.eval()


if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')  
    torch.cuda.set_device(local_rank)
    path = [f"/mnt/nas/shared/datasets/voices/AItts/AItts{i}/data_with_pinyinIPA.csv" for i in range(2,3)]
    dirctory =[f"/mnt/nas/shared/datasets/voices/AItts/AItts{i}/melspec" for i in range(2,3)]
    dirctory=["../AItts2mel/melspec"]
    path=["../AItts2.csv"]
    data = VariableLengthMelDataset(
        directories=dirctory, paths=path,cache=True
    )
    #data,_=random_split(data,[1000,len(data)-1000])
    sampler = DistributedSampler(data)
    data = DataLoader(data, batch_size=25, sampler=sampler)
    train(data, local_rank, epochs=400)
