import toml
import torch.optim as optim
from dataset import TTSDataset
from Glow_TTS.model import GlowTTS
from Vits.model import Vits
from torch.utils.data import DataLoader
model_set={
    "Glow_TTS" : GlowTTS,
    "Vits" : Vits
}

class Load_config:
    @staticmethod
    def load_config_toml(path):
        with open(path, 'r') as f:
            return toml.load(f)
    @staticmethod
    def load_model(config):
        model_config=config['model']
        return model_set[model_config['model_name']](**model_config)

    @staticmethod
    def load_optimizer(model,config):
        optim_config=config['optimizer']
        optim_type=optim_config['type']
        optimizer_class=getattr(optim,optim_type)
        return optimizer_class(model.parameters(),**optim_config['params'])

    @staticmethod
    def load_scheduler(optimizer,config):
        sched_config=config['shceduler']
        sched_type=sched_config['type']
        scheduler_class=getattr(optim.lr_scheduler,sched_type)
        return scheduler_class(optimizer,**sched_config['params'])

    @staticmethod
    def load_dataloader(config):
        dataset = TTSDataset(
            mandarin_file=config['train']['mandarin_file'],
            cantonese_file=config['train']['cantonese_file'],
            root_path=config['train']['root_path'],
            mandarin_num=config['train']['mandarin_num'],
            cantonese_num=config['train']['cantonese_num']
        )
        # 创建数据加载器
        loader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            **config['dataloader']
        )
        return loader