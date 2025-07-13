from config import GlowTTSConfig
from model import GlowTTS
from trainer import GlowTTSTrainer

def main():
    # 配置
    config = GlowTTSConfig(
        num_chars=61,
        out_channels=80,

        # 编码器参数
        encoder_type="rel_pos_transformer",
        encoder_params={
            "kernel_size": 3,
            "dropout_p": 0.1,
            "num_layers": 6,     # 从 6 增加到 12
            "num_heads": 2,       # 从 2 增加到 8
            "hidden_channels_ffn": 768,  # 从 768 增加到 1024
            "input_length": None,
        },

        # 编码器隐藏层 - 这些是分开的参数
        hidden_channels_enc=192,  # 从 192 
        hidden_channels_dec=192,  # 从 192 
        hidden_channels_dp=256,   # 从 256

        # Flow 参数
        num_flow_blocks_dec=12,   # 从 12 增加到 16
        num_block_layers=4,       # 从 4 增加到 6

        # 训练参数
        csv_file="data.csv",
        root_path="data",  # 假设数据存储在这个路径下
        epochs=100000,
        data_dep_init_steps=500,
        batch_size=1,
        lr=1e-3, 
        grad_clip=5.0,
        print_step=500,
        save_step=5000,
        run_eval=False,
        scheduler_after_epoch=False,  # NoamLR 按步调度
        optimizer="RAdam",
        optimizer_params={"betas": [0.9, 0.998], "weight_decay": 1e-6},
        lr_scheduler="NoamLR",
        lr_scheduler_params={"warmup_steps": 4000}
    )
    
    # 模型
    model = GlowTTS(config)
    
    # 训练器
    trainer = GlowTTSTrainer(
        model=model,
        config=config,
        output_path="./outputs"
    )
    
    # 开始训练
    # trainer.fit()
    trainer.fit_from_checkpoint("outputs/checkpoint_step_9999.pth")  # 从检查点恢复训练

if __name__ == "__main__":
    main()