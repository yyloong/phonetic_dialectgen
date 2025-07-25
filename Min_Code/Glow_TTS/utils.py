import numpy as np
import math
import torch
from torch.nn import functional as F


def Glow_TTS_loss(
    z,
    means,
    scales,
    log_det,
    y_lengths,
    o_dur_log,
    o_attn_dur,
    x_lengths,
):
    constant_factor = 0.5 * math.log(2 * math.pi)
    return_dict = {}
    # flow loss - neg log likelihood
    pz = torch.sum(scales) + 0.5 * torch.sum(torch.exp(-2 * scales) * (z - means) ** 2)
    log_mle = constant_factor + (pz - torch.sum(log_det)) / (
        torch.sum(y_lengths) * z.shape[2]
    )
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


def compute_outputs(attn, o_mean, o_log_scale, x_mask):
    """Compute and format the mode outputs with the given alignment map"""
    y_mean = torch.matmul(
        attn.squeeze(1).transpose(1, 2), o_mean.transpose(1, 2)
    ).transpose(
        1, 2
    )  # [b, t', t], [b, t, d] -> [b, d, t']
    y_log_scale = torch.matmul(
        attn.squeeze(1).transpose(1, 2), o_log_scale.transpose(1, 2)
    ).transpose(
        1, 2
    )  # [b, t', t], [b, t, d] -> [b, d, t']
    # compute total duration with adjustment
    o_attn_dur = torch.log(1 + torch.sum(attn, -1)) * x_mask
    return y_mean, y_log_scale, o_attn_dur



# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
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
    seq_range = torch.arange(
        max_len, dtype=sequence_length.dtype, device=sequence_length.device
    )
    # B x T_max
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    """
    Shapes:
        - duration: :math:`[B, T_en]`
        - mask: :math:'[B, T_en, T_de]`
        - path: :math:`[B, T_en, T_de]`
    """
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def maximum_path(value, mask, max_neg_val=None):
    """
    Monotonic alignment search algorithm
    Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    if max_neg_val is None:
        max_neg_val = -np.inf  # Patch for Sphinx complaint
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[
            :, :-1
        ]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path


class KeepAverage:
    def __init__(self):
        self.avg_values = {}
        self.iters = {}

    def __getitem__(self, key):
        return self.avg_values[key]

    def items(self):
        return self.avg_values.items()

    def add_value(self, name, init_val=0, init_iter=0):
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name, value, weighted_avg=False):
        if name not in self.avg_values:
            # add value if not exist before
            self.add_value(name, init_val=value)
        else:
            # else update existing value
            if weighted_avg:
                self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
                self.iters[name] += 1
            else:
                self.avg_values[name] = self.avg_values[name] * self.iters[name] + value
                self.iters[name] += 1
                self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict):
        for key, value in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict):
        for key, value in value_dict.items():
            self.update_value(key, value)
