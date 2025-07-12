import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
from model import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss

def train(x,mel):
    # 模型
    net_g = SynthesizerTrn(
        n_vocab=1024,
        spec_channels=80,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        

    ).cuda()
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()

    # 优化器
    optim_g = optim.AdamW(
        net_g.parameters(),
        lr=2e-4,
        betas=[0.8, 0.99],
        eps=12-9
    )
    optim_d = optim.AdamW(
        net_g.parameters(),
        lr=2e-4,
        betas=[0.8, 0.99],
        eps=12-9
    )

    # 训练循环
    net_g.train()
    net_d.train()
    for epoch in range(10000):
        x, x_lengths = x.cuda(), x_lengths.cuda()
        mel, mel_lengths = mel.cuda(), mel_lengths.cuda()
        # 生成器前向
        gen_mel, l_length, attn, z, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
            x, x_lengths, mel, mel_lengths
        )
        # 判别器损失
        mel_d_r, mel_d_g, _, _ = net_d(mel, gen_mel.detach())
        loss_disc, _, _ = discriminator_loss(mel_d_r, mel_d_g)

        optim_d.zero_grad()
        loss_disc.backward()
        optim_d.step()

        y_d_hat_r, mel_d_g, fmap_r, fmap_g = net_d(mel, gen_mel)
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(mel, gen_mel) * 45
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * 1
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(mel_d_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        optim_g.zero_grad()
        loss_gen_all.backward()
        optim_g.step()

if __name__ == "__main__":
    train()
