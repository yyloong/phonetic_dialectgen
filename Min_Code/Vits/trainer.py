import torch
from torch.nn import functional as F
from . import commons
from .losses import kl_loss
from Trainer import Trainers


class VitsTrainer(Trainers):

    def __init__(self, config,load_checkpoint,save_checkpoint,checkpoint_path=None):
        super().__init__(config,load_checkpoint,save_checkpoint,checkpoint_path)

    def run(self):

        if self.scheduler:
            self.scheduler.train()
        for epoch in range(self.epochs_done,self.epochs):
            for batch in self.train_loader:
                self.total_steps_done+=1
                # 生成器前向
                text_input = batch["token_ids"].to(self.device)
                text_lengths = batch["token_ids_lengths"].to(self.device)
                mel_input = batch["mel_input"].to(self.device)
                mel_lengths = batch["mel_lengths"].to(self.device)
                (
                    gen_mel,
                    l_length,
                    ids_slice,
                    z_mask,
                    (z_p, m_p, logs_p,logs_q),
                ) = self.model(text_input, text_lengths, mel_input, mel_lengths)
                mel_slice = commons.slice_segments(mel_input, ids_slice, 64)
                loss_dur = torch.sum(l_length.float())*3
                loss_mel = F.l1_loss(mel_slice, gen_mel) * 45
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                loss_gen_all = loss_mel + loss_dur + loss_kl#+loss_infer
                if self.total_steps_done% self.print_step == 0:
                    print("epoch:",epoch)
                    print("step:", self.total_steps_done)
                    print(
                        f"loss_gen_all:{loss_gen_all.item()}, loss_mel:{loss_mel.item()}, loss_dur:{loss_dur.item()}, loss_kl:{loss_kl.item()}"
                    )
                if self.total_steps_done%self.logs_step == 0:
                    self.logs_writer.add_scalar('loss',loss_gen_all.item(),self.total_steps_done)
                    self.logs_writer.add_scalar('loss_mel',loss_mel.item(),self.total_steps_done)
                    self.logs_writer.add_scalar('loss_kl',loss_kl.item(),self.total_steps_done)
                    self.logs_writer.add_scalar('loss_dur',loss_dur.item(),self.total_steps_done)
                    self.logs_writer.add_scalar('learning_rate',self.optimizer.param_groups[0]['lr'],self.total_steps_done)
                self.optimizer.zero_grad()
                loss_gen_all.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                # Save checkpoint
                if (self.total_steps_done + 1) % self.save_step == 0:
                    self.save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.total_steps_done,
                        self.epochs_done,
                        self.config,
                        self.config['train']['output_path'],
                    )
            self.epochs_done = epoch
        self.model.eval()
