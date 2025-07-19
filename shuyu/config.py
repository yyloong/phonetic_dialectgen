from dataclasses import dataclass, field


@dataclass
class GlowTTSConfig:
    """Defines parameters for GlowTTS model.

    Args:
        model(str):
            Model name used for selecting the right model at initialization. Defaults to `glow_tts`.
        num_chars (int):
            Number of characters in the vocabulary. It is used to define the input size of the encoder
            and the output size of the decoder. Defaults to 47.
        encoder_type (str):
            Encoder module type. Possible values are`["rel_pos_transformer", "gated_conv", "residual_conv_bn", "time_depth_separable"]`
            Check `layers.encoder` for more details. Defaults to `rel_pos_transformers` as in the original paper.
        encoder_params (dict):
            Parameters used to define the encoder network. Look at `layers.encoder` for more details.
            Defaults to `{"kernel_size": 3, "dropout_p": 0.1, "num_layers": 6, "num_heads": 2, "hidden_channels_ffn": 768}`
        use_encoder_prenet (bool):
            enable / disable the use of a prenet for the encoder. Defaults to True.
        hidden_channels_enc (int):
            Number of base hidden channels used by the encoder network. It defines the input and the output channel sizes,
            and for some encoder types internal hidden channels sizes too. Defaults to 192.
        hidden_channels_dec (int):
            Number of base hidden channels used by the decoder WaveNet network. Defaults to 192 as in the original work.
        hidden_channels_dp (int):
            Number of layer channels of the duration predictor network. Defaults to 256 as in the original work.
        dropout_p_dp (float):
            Dropout rate for the duration predictor. Defaults to 0.1.
        dropout_p_dec (float):
            Dropout rate for decoder. Defaults to 0.1.
        mean_only (bool):
            If True, encoder only computes mean value and uses constant variance for each time step. Defaults to true.
        out_channels (int):
            Number of channels of the model output tensor. Defaults to 80.
        num_flow_blocks_dec (int):
            Number of decoder blocks. Defaults to 12.
        kernel_size_dec (int):
            Decoder kernel size. Defaults to 5
        dilation_rate (int):
            Rate to increase dilation by each layer in a decoder block. Defaults to 1.
        num_block_layers (int):
            Number of decoder layers in each decoder block.  Defaults to 4.
        c_in_channels (int):
            Number of speaker embedding channels. It is set to 512 if embeddings are learned. Defaults to 0.
        num_splits (int):
            Number of split levels in inversible conv1x1 operation. Defaults to 4.
        num_squeeze (int):
            Number of squeeze levels. When squeezing channels increases and time steps reduces by the factor
            'num_squeeze'. Defaults to 2.
        sigmoid_scale (bool):
            enable/disable sigmoid scaling in decoder. Defaults to False.
        data_dep_init_steps (int):
            Number of steps used for computing normalization parameters at the beginning of the training. GlowTTS uses
            Activation Normalization that pre-computes normalization stats at the beginning and use the same values
            for the rest. Defaults to 10.
        epochs (int):
            Number of training epochs. Defaults to 10.
        batch_size (int):
            Batch size used for training. Defaults to 32.
        print_step (int):
            Print training progress every `print_step` steps. Defaults to 100.
        save_step (int):
            Save model every `save_step` steps. Defaults to 1000.
        run_eval (bool):
            Run evaluation every `save_step` steps. Defaults to True.
        csv_file (str):
            Path to the CSV file with training data. Defaults to `data.csv`.
        root_path (str):
            Path to the root directory with training data. Defaults to `data`.
        test_csv_file (str):
            Path to the CSV file with test data. Defaults to `test_data.csv`.
        test_root_path (str):
            Path to the root directory with test data. Defaults to `test_data`.
        inference_noise_scale (float):
            Variance used for sampling the random noise added to the decoder's input at inference. Defaults to 0.33.
        length_scale (float):
            Multiply the predicted durations with this value to change the speech speed. Defaults to 1.
        optimizer (str):
            Optimizer used for training. Defaults to `RAdam`.
        optimizer_params (dict):
            Parameters used to define the optimizer. Defaults to `{"betas": [0.9, 0.998], "weight_decay": 1e-6}`.
        use_scheduler (bool):
            If True, use a learning rate scheduler. Defaults to True.
        lr_scheduler (str):
            Learning rate scheduler used for training. Defaults to `NoamLR`.
        lr_scheduler_params (dict):
            Parameters used to define the learning rate scheduler. Defaults to `{"warmup_steps": 4000}`.
        scheduler_after_epoch (bool):
            If True, the learning rate scheduler is called after each epoch. Defaults to False.
        grad_clip (float):
            Gradient clipping value. Defaults to 5.0.
        lr (float):
            Initial learning rate. Defaults to `1e-3`.
    """

    model: str = "glow_tts"

    # model params
    num_chars: int = 38
    encoder_type: str = "rel_pos_transformer"
    encoder_params: dict = field(
        default_factory=lambda: {
            "kernel_size": 3,
            "dropout_p": 0.1,
            "num_layers": 6,
            "num_heads": 2,
            "hidden_channels_ffn": 768,
            "input_length": None,
        }
    )
    use_encoder_prenet: bool = True
    hidden_channels_enc: int = 192
    hidden_channels_dec: int = 192
    hidden_channels_dp: int = 256
    dropout_p_dp: float = 0.1
    dropout_p_dec: float = 0.05
    mean_only: bool = True
    out_channels: int = 80
    num_flow_blocks_dec: int = 12  # Flow 块数量
    kernel_size_dec: int = 5
    dilation_rate: int = 1
    num_block_layers: int = 4  # 每个 Flow 块的层数
    c_in_channels: int = 0
    num_splits: int = 4
    num_squeeze: int = 2
    sigmoid_scale: bool = False

    # training params
    data_dep_init_steps: int = 10
    epochs: int = 10
    batch_size: int = 32
    print_step: int = 100
    save_step: int = 1000
    run_eval: bool = True
    csv_file: str = "data.csv"
    root_path: str = "data"  # 假设数据存储在这个
    test_csv_file: str = "test_data.csv" 
    test_root_path: str = "test_data"

    # inference params
    inference_noise_scale: float = 0.33  # 🔥 温度参数
    length_scale: float = 1.0

    # optimizer parameters
    optimizer: str = "RAdam"
    optimizer_params: dict = field(
        default_factory=lambda: {"betas": [0.9, 0.998], "weight_decay": 1e-6}
    )
    use_scheduler: bool = True
    lr_scheduler: str = "NoamLR"
    lr_scheduler_params: dict = field(
        default_factory=lambda: {"warmup_steps": 4000}
    )
    scheduler_after_epoch: bool = False  # NoamLR 按步调度
    grad_clip: float = 5.0
    lr: float = 1e-3

    def __iter__(self):
        """使配置类可迭代"""
        return iter(self.__dict__.items())

    def keys(self):
        """返回所有配置键"""
        return self.__dict__.keys()

    def values(self):
        """返回所有配置值"""
        return self.__dict__.values()

    def items(self):
        """返回所有配置项"""
        return self.__dict__.items()
