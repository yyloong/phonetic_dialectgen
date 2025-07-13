import torch
import pdb
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
from model import SynthesizerTrn, Decoder
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss

def train(x,mel):
    # 模型
    net_g = SynthesizerTrn(
        n_vocab=14,
        spec_channels=80,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
    ).cuda()
    # 优化器
    optim_g = optim.AdamW(
        net_g.parameters(),
        lr=1e-4,
        betas=[0.8, 0.99],
        eps=12-9
    )
    
    net_g.train()
    x_lengths=torch.tensor([x.shape[-1]])
    mel_lengths=torch.tensor([mel.shape[-1]])
    x, x_lengths = x.cuda(), x_lengths.cuda()
    mel, mel_lengths = mel.cuda(), mel_lengths.cuda()
    for epoch in range(10000):
        # 生成器前向
        gen_mel, l_length, attn, z, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
            x, x_lengths, mel, mel_lengths
        )
        loss_dur = torch.sum(l_length.float())*5
        loss_mel = F.l1_loss(mel, gen_mel)* 45
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * 1
        loss_gen_all = loss_mel + loss_dur + loss_kl
        if epoch % 10 == 0:
            print("epoch:", epoch)
            print(f"loss_gen_all:{loss_gen_all.item()}, loss_mel:{loss_mel.item()}, loss_dur:{loss_dur.item()}, loss_kl:{loss_kl.item()}")
            print("gmel shape:", gen_mel.shape)
        if loss_mel.item() < 50 and loss_dur.item() <5:
            torch.save(gen_mel,"train_try.pt")
        optim_g.zero_grad()
        loss_gen_all.backward()
        optim_g.step()
    net_g.eval()
    with torch.no_grad():
        mel=net_g.infer(x,x_lengths)[0]
        print("mel shape:", mel.shape)
        torch.save(mel,"try.pt")

if __name__ == "__main__":
    mel=torch.load('/home/u-longyy/week2/999.pt').reshape(1,80,-1)
    x=torch.arange(0,14).reshape(1,-1)
    print("x shape:",x)
    print("mel shape:",mel.shape)
    train(x,mel)
