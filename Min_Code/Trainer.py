from load_config import Load_config
import os
from torch.utils.tensorboard import SummaryWriter

class Trainers:
    def __init__(self, config, load_checkpoint, save_checkpoint, checkpoint_path=None):
        self.device = config['train']['device']
        print(f"train by :{self.device}")
        self.model = Load_config.load_model(config).to(self.device)
        self.config = config
        self.logs_step = config['train']['logs_step'] # 定义多少步记录一次log到tensorboard
        self.load_checkpoint = load_checkpoint
        self.save_checkpoint = save_checkpoint
        self.optimizer = Load_config.load_optimizer(self.model, config)
        self.logs_writer = SummaryWriter(config['train']['tensorboard_logs_dir'])
        if config['train']['use_scheduler']:
            self.scheduler = Load_config.load_scheduler(self.optimizer, config)
        else:
            self.scheduler = None
        # Data loaders
        self.train_loader = Load_config.load_dataloader(config)
        self.total_steps_done = 0
        self.epochs_done = 0

        if checkpoint_path:
            #  config from checkpoint
            (
                self.model,
                self.optimizer,
                self.scheduler,
                self.total_steps_done,
                self.epochs_done,
                config,
            ) = self.load_checkpoint(
                self.model, self.optimizer, self.scheduler, checkpoint_path
            )

        for key, value in config['train'].items():
            setattr(self, key, value)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def run(self):
        raise NotImplementedError
