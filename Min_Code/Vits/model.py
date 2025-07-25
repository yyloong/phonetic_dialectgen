import torch
import math
import torch
from . import commons
from . import monotonic_align
from torch import nn
from .layers import (
    TextEncoder,
    Decoder,
    DurationPredictor,
    PosteriorEncoder,
    ResidualCouplingBlock,
)


class Vits(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,**model_config):

        super().__init__()

        self.enc_p = TextEncoder(**model_config['TextEncoder'])

        self.dec = Decoder(**model_config['Decoder'])

        self.enc_q = PosteriorEncoder(**model_config['PosteriorEncoder'])

        self.flow = ResidualCouplingBlock(**model_config['ResidualCouplingBlock'])

        self.dp = DurationPredictor(**model_config['DurationPredictor'])

    def forward(self, x, x_lengths, y, y_lengths):

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)
        z_p = self.flow(z, y_mask)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=None)
        l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, 64)
        o = self.dec(z_slice)
        return (
            o,
            l_length,
            ids_slice,
            y_mask,
            (z_p, m_p, logs_p,logs_q),
        )

    def inference(
        self,
        x,
        x_lengths,
        noise_scale=1,
        length_scale=1,
        max_len=None,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        logw = self.dp(x, x_mask)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, reverse=True)
        output = self.dec((z * y_mask)[:, :, :max_len])
        return {'model_outputs':output.transpose(1,2)}    #为了和Glow-TTS统一格式
