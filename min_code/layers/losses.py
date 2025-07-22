import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional
import librosa

from layers.ssim import SSIMLoss as _SSIMLoss

class TorchSTFT(nn.Module):  # pylint: disable=abstract-method
    """Some of the audio processing funtions using Torch for faster batch processing.

    Args:

        n_fft (int):
            FFT window size for STFT.

        hop_length (int):
            number of frames between STFT columns.

        win_length (int, optional):
            STFT window length.

        pad_wav (bool, optional):
            If True pad the audio with (n_fft - hop_length) / 2). Defaults to False.

        window (str, optional):
            The name of a function to create a window tensor that is applied/multiplied to each frame/window. Defaults to "hann_window"

        sample_rate (int, optional):
            target audio sampling rate. Defaults to None.

        mel_fmin (int, optional):
            minimum filter frequency for computing melspectrograms. Defaults to None.

        mel_fmax (int, optional):
            maximum filter frequency for computing melspectrograms. Defaults to None.

        n_mels (int, optional):
            number of melspectrogram dimensions. Defaults to None.

        use_mel (bool, optional):
            If True compute the melspectrograms otherwise. Defaults to False.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to False.

        spec_gain (float, optional):
            gain applied when converting amplitude to DB. Defaults to 1.0.

        power (float, optional):
            Exponent for the magnitude spectrogram, e.g., 1 for energy, 2 for power, etc.  Defaults to None.

        use_htk (bool, optional):
            Use HTK formula in mel filter instead of Slaney.

        mel_norm (None, 'slaney', or number, optional):
            If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization).

            If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
            See `librosa.util.normalize` for a full description of supported norm values
            (including `+-np.inf`).

            Otherwise, leave all the triangles aiming for a peak value of 1.0. Defaults to "slaney".
    """

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        pad_wav=False,
        window="hann_window",
        sample_rate=None,
        mel_fmin=0,
        mel_fmax=None,
        n_mels=80,
        use_mel=False,
        do_amp_to_db=False,
        spec_gain=1.0,
        power=None,
        use_htk=False,
        mel_norm="slaney",
        normalized=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad_wav = pad_wav
        self.sample_rate = sample_rate
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.n_mels = n_mels
        self.use_mel = use_mel
        self.do_amp_to_db = do_amp_to_db
        self.spec_gain = spec_gain
        self.power = power
        self.use_htk = use_htk
        self.mel_norm = mel_norm
        self.window = nn.Parameter(getattr(torch, window)(win_length), requires_grad=False)
        self.mel_basis = None
        self.normalized = normalized
        if use_mel:
            self._build_mel_basis()

    def __call__(self, x):
        """Compute spectrogram frames by torch based stft.

        Args:
            x (Tensor): input waveform

        Returns:
            Tensor: spectrogram frames.

        Shapes:
            x: [B x T] or [:math:`[B, 1, T]`]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if self.pad_wav:
            padding = int((self.n_fft - self.hop_length) / 2)
            x = torch.nn.functional.pad(x, (padding, padding), mode="reflect")
        # B x D x T x 2
        o = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            center=True,
            pad_mode="reflect",  # compatible with audio.py
            normalized=self.normalized,
            onesided=True,
            return_complex=False,
        )
        M = o[:, :, :, 0]
        P = o[:, :, :, 1]
        S = torch.sqrt(torch.clamp(M**2 + P**2, min=1e-8))

        if self.power is not None:
            S = S**self.power

        if self.use_mel:
            S = torch.matmul(self.mel_basis.to(x), S)
        if self.do_amp_to_db:
            S = self._amp_to_db(S, spec_gain=self.spec_gain)
        return S

    def _build_mel_basis(self):
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
            htk=self.use_htk,
            norm=self.mel_norm,
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()

    @staticmethod
    def _amp_to_db(x, spec_gain=1.0):
        return torch.log(torch.clamp(x, min=1e-5) * spec_gain)

    @staticmethod
    def _db_to_amp(x, spec_gain=1.0):
        return torch.exp(x) / spec_gain

def sequence_mask(sequence_length, max_len=None):
    """Create a sequence mask for filtering padding in a sequence tensor.

    Args:
        sequence_length (torch.tensor): Sequence lengths.
        max_len (int, Optional): Maximum sequence length. Defaults to None.

    Shapes:
        - mask: :math:`[B, T_max]`
    """
    if max_len is None:
        max_len = sequence_length.max()
    seq_range = torch.arange(max_len, dtype=sequence_length.dtype, device=sequence_length.device)
    # B x T_max
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)

# pylint: disable=abstract-method
# relates https://github.com/pytorch/pytorch/issues/42305
class L1LossMasked(nn.Module):
    def __init__(self, seq_len_norm):
        super().__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            x: B x T X D
            target: B x T x D
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.l1_loss(x * mask, target * mask, reduction="none")
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            loss = functional.l1_loss(x * mask, target * mask, reduction="sum")
            loss = loss / mask.sum()
        return loss


class MSELossMasked(nn.Module):
    def __init__(self, seq_len_norm):
        super().__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            - x: :math:`[B, T, D]`
            - target: :math:`[B, T, D]`
            - length: :math:`B`
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.mse_loss(x * mask, target * mask, reduction="none")
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            loss = functional.mse_loss(x * mask, target * mask, reduction="sum")
            loss = loss / mask.sum()
        return loss


def sample_wise_min_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Min-Max normalize tensor through first dimension
    Shapes:
        - x: :math:`[B, D1, D2]`
        - m: :math:`[B, D1, 1]`
    """
    maximum = torch.amax(x.masked_fill(~mask, 0), dim=(1, 2), keepdim=True)
    minimum = torch.amin(x.masked_fill(~mask, np.inf), dim=(1, 2), keepdim=True)
    return (x - minimum) / (maximum - minimum + 1e-8)


class SSIMLoss(torch.nn.Module):
    """SSIM loss as (1 - SSIM)
    SSIM is explained here https://en.wikipedia.org/wiki/Structural_similarity
    """

    def __init__(self):
        super().__init__()
        self.loss_func = _SSIMLoss()

    def forward(self, y_hat, y, length):
        """
        Args:
            y_hat (tensor): model prediction values.
            y (tensor): target values.
            length (tensor): length of each sample in a batch for masking.

        Shapes:
            y_hat: B x T X D
            y: B x T x D
            length: B

         Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        mask = sequence_mask(sequence_length=length, max_len=y.size(1)).unsqueeze(2)
        y_norm = sample_wise_min_max(y, mask)
        y_hat_norm = sample_wise_min_max(y_hat, mask)
        ssim_loss = self.loss_func((y_norm * mask).unsqueeze(1), (y_hat_norm * mask).unsqueeze(1))

        if ssim_loss.item() > 1.0:
            print(f" > SSIM loss is out-of-range {ssim_loss.item()}, setting it 1.0")
            ssim_loss = torch.tensor(1.0, device=ssim_loss.device)

        if ssim_loss.item() < 0.0:
            print(f" > SSIM loss is out-of-range {ssim_loss.item()}, setting it 0.0")
            ssim_loss = torch.tensor(0.0, device=ssim_loss.device)

        return ssim_loss


class AttentionEntropyLoss(nn.Module):
    # pylint: disable=R0201
    def forward(self, align):
        """
        Forces attention to be more decisive by penalizing
        soft attention weights
        """
        entropy = torch.distributions.Categorical(probs=align).entropy()
        loss = (entropy / np.log(align.shape[1])).mean()
        return loss


class BCELossMasked(nn.Module):
    """BCE loss with masking.

    Used mainly for stopnet in autoregressive models.

    Args:
        pos_weight (float): weight for positive samples. If set < 1, penalize early stopping. Defaults to None.
    """

    def __init__(self, pos_weight: float = None):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            x: B x T
            target: B x T
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        target.requires_grad = False
        if length is not None:
            # mask: (batch, max_len, 1)
            mask = sequence_mask(sequence_length=length, max_len=target.size(1))
            num_items = mask.sum()
            loss = functional.binary_cross_entropy_with_logits(
                x.masked_select(mask),
                target.masked_select(mask),
                pos_weight=self.pos_weight.to(x.device),
                reduction="sum",
            )
        else:
            loss = functional.binary_cross_entropy_with_logits(
                x, target, pos_weight=self.pos_weight.to(x.device), reduction="sum"
            )
            num_items = torch.numel(x)
        loss = loss / num_items
        return loss


class DifferentialSpectralLoss(nn.Module):
    """Differential Spectral Loss
    https://arxiv.org/ftp/arxiv/papers/1909/1909.10302.pdf"""

    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, x, target, length=None):
        """
         Shapes:
            x: B x T
            target: B x T
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        x_diff = x[:, 1:] - x[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        if length is None:
            return self.loss_func(x_diff, target_diff)
        return self.loss_func(x_diff, target_diff, length - 1)


class GuidedAttentionLoss(torch.nn.Module):
    def __init__(self, sigma=0.4):
        super().__init__()
        self.sigma = sigma

    def _make_ga_masks(self, ilens, olens):
        B = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        ga_masks = torch.zeros((B, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            ga_masks[idx, :olen, :ilen] = self._make_ga_mask(ilen, olen, self.sigma)
        return ga_masks

    def forward(self, att_ws, ilens, olens):
        ga_masks = self._make_ga_masks(ilens, olens).to(att_ws.device)
        seq_masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = ga_masks * att_ws
        loss = torch.mean(losses.masked_select(seq_masks))
        return loss

    @staticmethod
    def _make_ga_mask(ilen, olen, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(olen).to(olen), torch.arange(ilen).to(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma**2)))

    @staticmethod
    def _make_masks(ilens, olens):
        in_masks = sequence_mask(ilens)
        out_masks = sequence_mask(olens)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)


class Huber(nn.Module):
    # pylint: disable=R0201
    def forward(self, x, y, length=None):
        """
        Shapes:
            x: B x T
            y: B x T
            length: B
        """
        mask = sequence_mask(sequence_length=length, max_len=y.size(1)).unsqueeze(2).float()
        return torch.nn.functional.smooth_l1_loss(x * mask, y * mask, reduction="sum") / mask.sum()


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = torch.nn.functional.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss = total_loss + loss

        total_loss = total_loss / attn_logprob.shape[0]
        return total_loss


########################
# MODEL LOSS LAYERS
########################
class GlowTTSLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.constant_factor = 0.5 * math.log(2 * math.pi)

    def forward(self, z, means, scales, log_det, y_lengths, o_dur_log, o_attn_dur, x_lengths):
        return_dict = {}
        # flow loss - neg log likelihood
        pz = torch.sum(scales) + 0.5 * torch.sum(torch.exp(-2 * scales) * (z - means) ** 2)
        log_mle = self.constant_factor + (pz - torch.sum(log_det)) / (torch.sum(y_lengths) * z.shape[2])
        # duration loss - MSE
        loss_dur = torch.sum((o_dur_log - o_attn_dur) ** 2) / torch.sum(x_lengths)
        # duration loss - huber loss
        # loss_dur = torch.nn.functional.smooth_l1_loss(o_dur_log, o_attn_dur, reduction="sum") / torch.sum(x_lengths)
        return_dict["loss"] = log_mle + loss_dur
        return_dict["log_mle"] = log_mle
        return_dict["loss_dur"] = loss_dur

        # check if any loss is NaN
        for key, loss in return_dict.items():
            if torch.isnan(loss):
                raise RuntimeError(f" [!] NaN loss with {key}.")
        return return_dict


def mse_loss_custom(x, y):
    """MSE loss using the torch back-end without reduction.
    It uses less VRAM than the raw code"""
    expanded_x, expanded_y = torch.broadcast_tensors(x, y)
    return torch._C._nn.mse_loss(expanded_x, expanded_y, 0)  # pylint: disable=protected-access, c-extension-no-member