import torch
from model import GlowTTS


def load_model_from_checkpoint(checkpoint_path, config=None):
    """从检查点加载模型进行推理"""
    # 1. 加载检查点
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    # 2. 获取配置（如果没有提供的话）
    if config is None:
        config = checkpoint.get("config", None)
        if config is None:
            raise ValueError("检查点中没有配置信息，请手动提供配置")
    # config.inference_noise_scale = 0.0  # 推理时不使用噪声缩放
    # 3. 创建模型
    model = GlowTTS(config)
    # 4. 加载模型权重
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    # 5. 设置为评估模式
    model.eval()
    print(f"✅ 模型已从检查点加载: {checkpoint_path}")
    print(
        f"📊 检查点信息: 步骤 {checkpoint.get('total_steps_done', 'N/A')}, "
        f"Epoch {checkpoint.get('epochs_done', 'N/A')}, "
        f"损失: {checkpoint.get('best_loss', 'N/A')}"
    )
    return model, config