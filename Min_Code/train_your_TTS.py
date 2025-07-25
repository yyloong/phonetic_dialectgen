from Glow_TTS.trainer import GlowTTSTrainer
from Vits.trainer import VitsTrainer
from load_config import Load_config
from load_save_checkpoint import load_checkpoint, save_checkpoint
import argparse
model_trainer = {
    'Glow-TTS': GlowTTSTrainer,
    'Vits' : VitsTrainer
}

def main(args):
    trainer = model_trainer[args.model](
        Load_config.load_config_toml(args.config_path),
        load_checkpoint,
        save_checkpoint,
        args.checkpoint_path,
    )
    trainer.run()


if __name__ == "__main__":
    '''从命令行接受参数'''
    parser = argparse.ArgumentParser(description="Train a model with TOML config")
    parser.add_argument(
        "--config_path",
        default="Tomls_config/Glow_TTS.toml",
        help="Path to TOML config file",
    )

    parser.add_argument("--model", default="Glow-TTS", help="The model you want to train")

    parser.add_argument(
        "--checkpoint_path", default=None, help="Path to your checkpont"
    )

    args = parser.parse_args()

    main(args)
