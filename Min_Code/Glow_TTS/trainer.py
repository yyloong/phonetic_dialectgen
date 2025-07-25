import torch
from load_config import Load_config
from typing import Dict, Tuple
from .utils import Glow_TTS_loss
from .utils import KeepAverage
from Trainer import Trainers


class GlowTTSTrainer(Trainers):
    """Simplified GlowTTS Trainer - Single GPU training only"""

    def __init__(self, config,load_checkpoint,save_checkpoint,checkpoint_path=None):
        super().__init__(config,load_checkpoint,save_checkpoint,checkpoint_path)


    def unlock_act_norm_layers(self):
        """Unlock activation normalization layers for data depended initalization."""
        for f in self.model.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(True)

    def lock_act_norm_layers(self):
        """Lock activation normalization layers."""
        for f in self.model.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(False)

    def train_step(self, batch: Dict) -> Tuple[Dict, Dict]:
        """Execute one training step"""
        # Directly manage data dependency initialization
        self.run_data_dep_init = self.total_steps_done < self.data_dep_init_steps

        # Forward pass and loss calculation
        text_input = batch["token_ids"].to(self.device)
        text_lengths = batch["token_ids_lengths"].to(self.device)
        mel_input = batch["mel_input"].to(self.device)
        mel_lengths = batch["mel_lengths"].to(self.device)

        if self.run_data_dep_init and self.model.training:
            # compute data-dependent initialization of activation norm layers
            self.unlock_act_norm_layers()
            with torch.no_grad():
                _ = self.model(
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
            outputs = self.model(
                text_input,
                text_lengths,
                mel_input,
                mel_lengths,
            )

            loss_dict = Glow_TTS_loss(
                outputs["z"].float(),
                outputs["y_mean"].float(),
                outputs["y_log_scale"].float(),
                outputs["logdet"].float(),
                mel_lengths,
                outputs["durations_log"].float(),
                outputs["total_durations_log"].float(),
                text_lengths,
            )
        # Update step count
        self.total_steps_done += 1

        # Skip backpropagation for data dependency initialization steps
        if not loss_dict:
            return {}

        # Backward pass
        loss = loss_dict["loss"]
        loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()

        # Detach loss dictionary
        loss_dict_detached = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }
        return loss_dict_detached

    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        torch.set_grad_enabled(True)

        print(f"🚀 Starting training Epoch {self.epochs_done}")

        # print(f"📊 Training data loader: {len(train_loader)} batches")
        for step, batch in enumerate(self.train_loader):
            # Execute training step
            loss_dict = self.train_step(batch)

            if not loss_dict:
                print(f" [!] Step {step} skipped (data dependency initialization)")
                continue

            # Update statistics
            self.keep_avg_train.update_values(
                {f"avg_{k}": v for k, v in loss_dict.items()}
            )

            # Print training progress
            if (step + 1) % self.print_step == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch:{self.epochs_done}")
                print(
                    f"🔄 Step {step+1}/{len(self.train_loader)} | "
                    f"Loss: {loss_dict['loss']:.4f} | "
                    f"Flow Loss: {loss_dict['log_mle']:.4f} | "
                    f"Duration Loss: {loss_dict['loss_dur']:.4f} | "
                    f"LR: {current_lr:.2e} | "
                )
            if self.total_steps_done%self.logs_step == 0:
                    self.logs_writer.add_scalar('loss',loss_dict['loss'],self.total_steps_done)
                    self.logs_writer.add_scalar('flow_loss',loss_dict['log_mle'],self.total_steps_done)
                    self.logs_writer.add_scalar('duration loss',loss_dict['loss_dur'],self.total_steps_done)
                    self.logs_writer.add_scalar('learning_rate',self.optimizer.param_groups[0]['lr'],self.total_steps_done)
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

        # Learning rate scheduling after epoch
        if self.scheduler is not None:
            scheduler_name = getattr(self.config, "lr_scheduler", "ExponentialLR")
            # NoamLR schedules by step, other schedulers by epoch
            if scheduler_name != "NoamLR" and getattr(
                self.config, "scheduler_after_epoch", True
            ):
                self.scheduler.step()

        print(f"✅ Epoch {self.epochs_done} completed")

        # Print average statistics
        avg_stats = self.keep_avg_train.avg_values
        print(
            f"📊 Average Loss: {avg_stats.get('avg_loss', 0):.4f} | "
            f"Average Flow Loss: {avg_stats.get('avg_log_mle', 0):.4f} | "
            f"Average Duration Loss: {avg_stats.get('avg_loss_dur', 0):.4f}"
        )

    def run(self):
        """Main training loop"""
        
        print("🎯 Starting GlowTTS training")

        for epoch in range(self.epochs_done, self.epochs):
            self.epochs_done = epoch

            # Reset statistics
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage()

            # Training
            self.train_epoch()

            print(f"🎊 Epoch {epoch} completed\n" + "=" * 50)

        print("🎉 Training completed!")
        print(f"🏆 Best loss: {self.best_loss:.4f}")
