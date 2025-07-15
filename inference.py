import torch
from config import GlowTTSConfig
from model import GlowTTS
from tokenizer import TTSTokenizer

def load_model_from_checkpoint(checkpoint_path, config=None):
    """从检查点加载模型进行推理"""
    
    # 1. 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 2. 获取配置（如果没有提供的话）
    if config is None:
        config = checkpoint.get('config', None)
        if config is None:
            raise ValueError("检查点中没有配置信息，请手动提供配置")
        
    # config.inference_noise_scale = 0.0  # 推理时不使用噪声缩放
    
    # 3. 创建模型
    model = GlowTTS(config)
    
    # 4. 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 5. 设置为评估模式
    model.eval()
    
    print(f"✅ 模型已从检查点加载: {checkpoint_path}")
    print(f"📊 检查点信息: 步骤 {checkpoint.get('total_steps_done', 'N/A')}, "
          f"Epoch {checkpoint.get('epochs_done', 'N/A')}, "
          f"损失: {checkpoint.get('best_loss', 'N/A')}")
    
    return model, config

def inference_example():
    """推理示例"""
    
    # 1. 加载模型
    checkpoint_path = "./outputs/checkpoint_step_107999.pth"  # 你的检查点路径
    # config = GlowTTSConfig(
    #     num_chars=100,
    #     out_channels=80,

    #     # 编码器参数
    #     encoder_type="rel_pos_transformer",
    #     encoder_params={
    #         "kernel_size": 3,
    #         "dropout_p": 0.1,
    #         "num_layers": 12,     # 从 6 增加到 12
    #         "num_heads": 8,       # 从 2 增加到 8
    #         "hidden_channels_ffn": 1024,  # 从 768 增加到 1024
    #         "input_length": None,
    #     },

    #     # 编码器隐藏层 - 这些是分开的参数
    #     hidden_channels_enc=256,  # 从 192 
    #     hidden_channels_dec=256,  # 从 192 
    #     hidden_channels_dp=256,   # 从 256

    #     # Flow 参数
    #     num_flow_blocks_dec=16,   # 从 12 增加到 16
    #     num_block_layers=6,       # 从 4 增加到 6

    #     # 训练参数
    #     epochs=5,
    #     data_dep_init_steps=2,
    #     batch_size=16,
    #     lr=5e-4, 
    #     grad_clip=5.0,
    #     print_step=10,
    #     save_step=1000,
    #     run_eval=True,
    #     scheduler_after_epoch=False,  # NoamLR 按步调度
    #     optimizer="RAdam",
    #     optimizer_params={"betas": [0.9, 0.998], "weight_decay": 1e-6},
    #     lr_scheduler="NoamLR",
    #     lr_scheduler_params={"warmup_steps": 4000}
    # )
    # model, _ = load_model_from_checkpoint(checkpoint_path, config)
    model, config = load_model_from_checkpoint(checkpoint_path)
    
    # 2. 准备输入文本
    text = "ye51 liaŋ51 thou55 thou55 kei215 ɕiau215 kou215 suŋ51 lɤ0 thaŋ35 kuo215 。"
    
    # 3. 文本预处理（需要根据你的tokenizer调整）
    tokenizer = TTSTokenizer()
    
    # 将文本转换为token序列
    token_ids = tokenizer(text)
    
    # 转换为tensor
    text_input = torch.LongTensor(token_ids).unsqueeze(0)  # [1, seq_len]
    text_lengths = torch.LongTensor([len(token_ids)])      # [1]
    
    # 4. 推理
    with torch.no_grad():
        # 生成语音
        outputs = model.inference(
            x=text_input,
            x_lengths=text_lengths
        )
        
        # 获取梅尔频谱
        mel_spectrogram = outputs["model_outputs"]  # [1, T, C]
        
    print(f"🎵 生成的梅尔频谱形状: {mel_spectrogram.shape}")
    
    return mel_spectrogram

if __name__ == "__main__":
    mel_output = inference_example()
    torch.save(mel_output, "mel_output.pth")
    print("✅ 推理完成，梅尔频谱已保存为 mel_output.pth")