import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict

from config import GlowTTSConfig
from layers.decoder import Decoder
from layers.encoder import Encoder
from utils import generate_path, maximum_path, sequence_mask
from torch.utils.data import DataLoader
from dataset import TTSDataset  # 假设有这个数据集类
from tokenizer import TTSTokenizer

class GlowTTS(nn.Module):
    """GlowTTS model.
    Examples:
        Init only model layers.

        >>> from TTS.tts.configs.glow_tts_config import GlowTTSConfig
        >>> from TTS.tts.models.glow_tts import GlowTTS
        >>> config = GlowTTSConfig(num_chars=2)
        >>> model = GlowTTS(config)

        Fully init a model ready for action. All the class attributes and class members
        (e.g Tokenizer, AudioProcessor, etc.). are initialized internally based on config values.

        >>> from TTS.tts.configs.glow_tts_config import GlowTTSConfig
        >>> from TTS.tts.models.glow_tts import GlowTTS
        >>> config = GlowTTSConfig()
        >>> model = GlowTTS.init_from_config(config, verbose=False)
    """

    def __init__(
        self,
        config: GlowTTSConfig
    ):
        super().__init__()
        # pass all config fields to `self`
        # for fewer code change
        for key, value in config.items():
            setattr(self, key, value)

        self.decoder_output_dim = config.out_channels

        self.run_data_dep_init = config.data_dep_init_steps > 0
        self.encoder = Encoder(
            self.num_chars,
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels_enc,
            hidden_channels_dp=self.hidden_channels_dp,
            encoder_type=self.encoder_type,
            encoder_params=self.encoder_params,
            mean_only=self.mean_only,
            use_prenet=self.use_encoder_prenet,
            dropout_p_dp=self.dropout_p_dp,
            c_in_channels=self.c_in_channels,
        )

        self.decoder = Decoder(
            self.out_channels,
            self.hidden_channels_dec,
            self.kernel_size_dec,
            self.dilation_rate,
            self.num_flow_blocks_dec,
            self.num_block_layers,
            dropout_p=self.dropout_p_dec,
            num_splits=self.num_splits,
            num_squeeze=self.num_squeeze,
            sigmoid_scale=self.sigmoid_scale,
            c_in_channels=self.c_in_channels,
        )

    @staticmethod
    def compute_outputs(attn, o_mean, o_log_scale, x_mask):
        """Compute and format the mode outputs with the given alignment map"""
        y_mean = torch.matmul(attn.squeeze(1).transpose(1, 2), o_mean.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        y_log_scale = torch.matmul(attn.squeeze(1).transpose(1, 2), o_log_scale.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        # compute total duration with adjustment
        o_attn_dur = torch.log(1 + torch.sum(attn, -1)) * x_mask
        return y_mean, y_log_scale, o_attn_dur

    def unlock_act_norm_layers(self):
        """Unlock activation normalization layers for data depended initalization."""
        for f in self.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(True)

    def lock_act_norm_layers(self):
        """Lock activation normalization layers."""
        for f in self.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(False)

    def forward(
        self, x, x_lengths, y, y_lengths=None
    ):
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
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x, x_lengths, g=None)
        # drop redisual frames wrt num_squeeze and set y_lengths.
        y, y_lengths, y_max_length, attn = self.preprocess(y, y_lengths, y_max_length, None)
        # create masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        # [B, 1, T_en, T_de]
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # decoder pass (without speaker embedding)
        z, logdet = self.decoder(y, y_mask, g=None, reverse=False)
        # find the alignment path
        with torch.no_grad():
            o_scale = torch.exp(-2 * o_log_scale)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - o_log_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 * (z**2))  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((o_mean * o_scale).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (o_mean**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(attn, o_mean, o_log_scale, x_mask)
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

    # @torch.no_grad()
    # def inference_with_MAS(
    #     self, x, x_lengths, y=None, y_lengths=None
    # ):
    #     """
    #     It's similar to the teacher forcing in Tacotron.
    #     It was proposed in: https://arxiv.org/abs/2104.05557

    #     Shapes:
    #         - x: :math:`[B, T]`      text token ids
    #         - x_lenghts: :math:`B`   lengths of input text sequences
    #         - y: :math:`[B, T, C]`   target mel-spectrogram
    #         - y_lengths: :math:`B`   length of target mel-spectrogram frames
    #     """
    #     y = y.transpose(1, 2)
    #     y_max_length = y.size(2)
    #     # embedding pass (without speaker embedding)
    #     o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x, x_lengths, g=None)
    #     # drop redisual frames wrt num_squeeze and set y_lengths.
    #     y, y_lengths, y_max_length, attn = self.preprocess(y, y_lengths, y_max_length, None)
    #     # create masks
    #     y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
    #     attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
    #     # decoder pass (without speaker embedding)
    #     z, logdet = self.decoder(y, y_mask, g=None, reverse=False)
    #     # find the alignment path between z and encoder output
    #     o_scale = torch.exp(-2 * o_log_scale)
    #     logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - o_log_scale, [1]).unsqueeze(-1)  # [b, t, 1]
    #     logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 * (z**2))  # [b, t, d] x [b, d, t'] = [b, t, t']
    #     logp3 = torch.matmul((o_mean * o_scale).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
    #     logp4 = torch.sum(-0.5 * (o_mean**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
    #     logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']
    #     attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

    #     y_mean, y_log_scale, o_attn_dur = self.compute_outputs(attn, o_mean, o_log_scale, x_mask)
    #     attn = attn.squeeze(1).permute(0, 2, 1)

    #     # get predited aligned distribution
    #     z = y_mean * y_mask

    #     # reverse the decoder and predict using the aligned distribution
    #     y, logdet = self.decoder(z, y_mask, g=None, reverse=True)
    #     outputs = {
    #         "model_outputs": z.transpose(1, 2),
    #         "logdet": logdet,
    #         "y_mean": y_mean.transpose(1, 2),
    #         "y_log_scale": y_log_scale.transpose(1, 2),
    #         "alignments": attn,
    #         "durations_log": o_dur_log.transpose(1, 2),
    #         "total_durations_log": o_attn_dur.transpose(1, 2),
    #     }
    #     return outputs

    @torch.no_grad()
    def decoder_inference(
        self, y, y_lengths=None
    ):
        """
        Shapes:
            - y: :math:`[B, C, T]`
            - y_lengths: :math:`B`
        """
        y_max_length = y.size(2)
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(y.dtype)
        # decoder pass (without speaker embedding)
        z, logdet = self.decoder(y, y_mask, g=None, reverse=False)
        # reverse decoder and predict
        y, logdet = self.decoder(z, y_mask, g=None, reverse=True)
        outputs = {}
        outputs["model_outputs"] = y
        outputs["logdet"] = logdet
        return outputs

    @torch.no_grad()
    def inference(
        self, x, x_lengths=None
    ):
        # embedding pass (without speaker embedding)
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x, x_lengths, g=None)
        # compute output durations
        w = (torch.exp(o_dur_log) - 1) * x_mask * self.length_scale
        w_ceil = torch.clamp_min(torch.ceil(w), 1)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = None
        # compute masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # compute attention mask
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(attn, o_mean, o_log_scale, x_mask)
        # add noise to the decoder input (inference noise scale is the temperature parameter)
        z = (y_mean + torch.exp(y_log_scale) * torch.randn_like(y_mean) * self.inference_noise_scale) * y_mask
        # decoder pass (without speaker embedding)
        y, logdet = self.decoder(z, y_mask, g=None, reverse=True)
        attn = attn.squeeze(1).permute(0, 2, 1)
        outputs = {
            "model_outputs": y.transpose(1, 2),   # mel_spectrograms [B, T, C]
            "logdet": logdet,
            "y_mean": y_mean.transpose(1, 2),
            "y_log_scale": y_log_scale.transpose(1, 2),
            "alignments": attn,
            "durations_log": o_dur_log.transpose(1, 2),
            "total_durations_log": o_attn_dur.transpose(1, 2),
        }
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module):
        """A single training step. Forward pass and loss computation. Run data depended initialization for the
        first `config.data_dep_init_steps` steps.

        Args:
            batch (dict): [description]
            criterion (nn.Module): [description]
        """
        text_input = batch["token_ids"]
        text_lengths = batch["token_ids_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]

        if self.run_data_dep_init and self.training:
            # compute data-dependent initialization of activation norm layers
            self.unlock_act_norm_layers()
            with torch.no_grad():
                _ = self.forward(
                    text_input,
                    text_lengths,
                    mel_input,
                    mel_lengths,
                )
            outputs = None
            loss_dict = None
            self.lock_act_norm_layers()
        else:
            # normal training step
            outputs = self.forward(
                text_input,
                text_lengths,
                mel_input,
                mel_lengths,
            )

            loss_dict = criterion(
                outputs["z"].float(),
                outputs["y_mean"].float(),
                outputs["y_log_scale"].float(),
                outputs["logdet"].float(),
                mel_lengths,
                outputs["durations_log"].float(),
                outputs["total_durations_log"].float(),
                text_lengths,
            )
        return outputs, loss_dict

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    def preprocess(self, y, y_lengths, y_max_length, attn=None):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.num_squeeze) * self.num_squeeze
            y = y[:, :, :y_max_length]
            if attn is not None:
                attn = attn[:, :, :, :y_max_length]
        y_lengths = torch.div(y_lengths, self.num_squeeze, rounding_mode="floor") * self.num_squeeze
        return y, y_lengths, y_max_length, attn

    def store_inverse(self):
        self.decoder.store_inverse()

    @staticmethod
    def get_criterion():
        return GlowTTSLoss()

    def get_data_loader(self, config, is_eval):
        """创建数据加载器"""        
        # 创建数据集
        dataset = TTSDataset(
            tokenizer=TTSTokenizer(),
            csv_file=config.csv_file,
            root_path=config.root_path
        )
        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=not is_eval,
            drop_last=not is_eval,
            collate_fn=dataset.collate_fn
        )
        return loader
    
    # def format_batch(self, batch: Dict) -> Dict:
    #     """Simplified batch formatting for single-speaker GlowTTS.

    #     Args:
    #         batch (Dict): Raw batch from dataloader

    #     Returns:
    #         Dict: Formatted batch for GlowTTS training
    #     """
    #     # 由 dataset.collate_fn 返回的 batch 已经是格式化好的数据
    
    #     return {
    #         "token_ids": batch["token_ids"],                    # 文本 token IDs
    #         "token_ids_lengths": batch["token_ids_lengths"],    # 文本长度
    #         "mel_input": batch["mel_input"],                    # 梅尔频谱  [B, C, T]
    #         "mel_lengths": batch["mel_lengths"]                 # 梅尔频谱长度
    #     }
    
    #     text_input = batch["token_ids"]           # 文本 token IDs
    #     text_lengths = batch["token_ids_lengths"] # 文本长度
    #     mel_input = batch["mel_input"]            # 梅尔频谱  [B, C, T]
    #     mel_lengths = batch["mel_lengths"]        # 梅尔频谱长度
        
    #     # 计算最大长度（用于 padding）
    #     max_text_length = torch.max(text_lengths.float())
    #     max_spec_length = torch.max(mel_lengths.float())
        
    #     # 从注意力掩码计算持续时间（如果有的话）
    #     durations = None
    #     if "attns" in batch and batch["attns"] is not None:
    #         attn_mask = batch["attns"]
    #         durations = torch.zeros(attn_mask.shape[0], attn_mask.shape[2])
    #         for idx, am in enumerate(attn_mask):
    #             # 计算原始持续时间
    #             c_idxs = am[:, : text_lengths[idx], : mel_lengths[idx]].max(1)[1]
    #             c_idxs, counts = torch.unique(c_idxs, return_counts=True)
    #             dur = torch.ones([text_lengths[idx]]).to(counts.dtype)
    #             dur[c_idxs] = counts
                
    #             # 平滑持续时间，确保总和等于梅尔频谱长度
    #             extra_frames = dur.sum() - mel_lengths[idx]
    #             if extra_frames > 0:
    #                 largest_idxs = torch.argsort(-dur)[:extra_frames]
    #                 dur[largest_idxs] -= 1
                
    #             durations[idx, : text_lengths[idx]] = dur
        
    #     # 返回 GlowTTS 需要的最小数据集
    #     return {
    #         "token_ids": text_input,
    #         "token_ids_lengths": text_lengths,
    #         "mel_input": mel_input,   # [B, C, T]
    #         "mel_lengths": mel_lengths,
    #         "durations": durations,
    #         "max_text_length": float(max_text_length),
    #         "max_spec_length": float(max_spec_length),
    #     }


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


if __name__ == "__main__":
    # Example usage
    config = GlowTTSConfig(num_chars=100, out_channels=80)
    tokenizer = TTSTokenizer()
    model = GlowTTS(config)
    print(model)

    # Dummy input
    x = torch.randint(0, 100, (2, 50))  # Batch of 2 sequences of length 50
    x_lengths = torch.tensor([50, 50])
    y = torch.randn(2, 80, 100)  # Batch of 2 mel spectrograms with 80 channels and length 100
    y_lengths = torch.tensor([100, 100])

    outputs = model.forward(x, x_lengths, y, y_lengths)
    print(outputs)