import math
import torch
from torch import nn
from .layers.decoder import Decoder
from .layers.encoder import Encoder
from .utils import generate_path, maximum_path, sequence_mask,compute_outputs

class GlowTTS(nn.Module):

    def __init__(self, **model_config):
        super().__init__()
        # pass all config fields to `self`
        # for fewer code change

        self.length_scale = model_config['length_scale']

        self.inference_noise_scale = model_config['inference_noise_scale']

        self.encoder = Encoder(**model_config['encoder'])

        self.decoder = Decoder(**model_config['decoder'])

    
    def forward(self, x, x_lengths, y, y_lengths=None):
        """
        Args:
            x (torch.Tensor):
                Input text sequence ids. :math:`[B, T_en]`

            x_lengths (torch.Tensor):
                Lengths of input text sequences. :math:`[B]`

            y (torch.Tensor):
                Target mel-spectrogram frames. :math:`[B, C_mel, T_de]`

            y_lengths (torch.Tensor):
                Lengths of target mel-spectrogram frames. :math:`[B]`

        Returns:
            Dict:
                - z: :math: `[B, T_de, C]`
                - logdet: :math:`B`
                - y_mean: :math:`[B, T_de, C]`
                - y_log_scale: :math:`[B, T_de, C]`
                - alignments: :math:`[B, T_en, T_de]`
                - durations_log: :math:`[B, T_en, 1]`
                - total_durations_log: :math:`[B, T_en, 1]`
        """
        # [B, C, T]
        y_max_length = y.size(2)
        # embedding pass (without speaker embedding)
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(
            x, x_lengths, g=None
        )
        # drop redisual frames wrt num_squeeze and set y_lengths.
        y, y_lengths, y_max_length, attn = self.decoder.preprocess(
            y, y_lengths, y_max_length, None
        )
        # create masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(
            x_mask.dtype
        )
        # [B, 1, T_en, T_de]
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # decoder pass (without speaker embedding)
        z, logdet = self.decoder(y, y_mask, g=None, reverse=False)
        # find the alignment path
        with torch.no_grad():
            o_scale = torch.exp(-2 * o_log_scale)
            logp1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - o_log_scale, [1]
            ).unsqueeze(
                -1
            )  # [b, t, 1]
            logp2 = torch.matmul(
                o_scale.transpose(1, 2), -0.5 * (z**2)
            )  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul(
                (o_mean * o_scale).transpose(1, 2), z
            )  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (o_mean**2) * o_scale, [1]).unsqueeze(
                -1
            )  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']
            attn = (
                maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
            )
        y_mean, y_log_scale, o_attn_dur = compute_outputs(
            attn, o_mean, o_log_scale, x_mask
        )
        attn = attn.squeeze(1).permute(0, 2, 1)
        outputs = {
            "z": z.transpose(1, 2),
            "logdet": logdet,
            "y_mean": y_mean.transpose(1, 2),
            "y_log_scale": y_log_scale.transpose(1, 2),
            "alignments": attn,
            "durations_log": o_dur_log.transpose(1, 2),
            "total_durations_log": o_attn_dur.transpose(1, 2),
        }
        return outputs

    @torch.no_grad()
    def inference(self, x, x_lengths=None):
        # embedding pass (without speaker embedding)
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(
            x, x_lengths, g=None
        )
        # compute output durations
        w = (torch.exp(o_dur_log) - 1) * x_mask * self.length_scale
        w_ceil = torch.clamp_min(torch.ceil(w), 1)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = None
        # compute masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # compute attention mask
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(
            1
        )
        y_mean, y_log_scale, o_attn_dur =compute_outputs(
            attn, o_mean, o_log_scale, x_mask
        )
        # add noise to the decoder input (inference noise scale is the temperature parameter)
        z = (
            y_mean
            + torch.exp(y_log_scale)
            * torch.randn_like(y_mean)
            * self.inference_noise_scale
        ) * y_mask
        # decoder pass (without speaker embedding)
        y, logdet = self.decoder(z, y_mask, g=None, reverse=True)
        attn = attn.squeeze(1).permute(0, 2, 1)
        outputs = {
            "model_outputs": y.transpose(1, 2),  # mel_spectrograms [B, T, C]
            "logdet": logdet,
            "y_mean": y_mean.transpose(1, 2),
            "y_log_scale": y_log_scale.transpose(1, 2),
            "alignments": attn,
            "durations_log": o_dur_log.transpose(1, 2),
            "total_durations_log": o_attn_dur.transpose(1, 2),
        }
        return outputs