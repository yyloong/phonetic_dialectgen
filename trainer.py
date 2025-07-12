import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from typing import Dict, Tuple
import time
from utils import KeepAverage
import os

def create_optimizer(model, config):
    """Create optimizer based on configuration"""
    optimizer_name = getattr(config, 'optimizer', 'Adam')
    lr = getattr(config, 'lr', 1e-3)
    optimizer_params = getattr(config, 'optimizer_params', {})
    
    # Get optimizer class
    if hasattr(torch.optim, optimizer_name):
        optimizer_class = getattr(torch.optim, optimizer_name)
    else:
        print(f"⚠️  Optimizer {optimizer_name} does not exist, using default Adam")
        optimizer_class = torch.optim.Adam
        optimizer_params = {"betas": [0.9, 0.999], "weight_decay": 0}
    
    try:
        optimizer = optimizer_class(model.parameters(), lr=lr, **optimizer_params)
        print(f"✅ Created optimizer: {optimizer_name}, LR: {lr}, Parameters: {optimizer_params}")
        return optimizer
    except Exception as e:
        print(f"❌ Failed to create optimizer: {e}, using default Adam")
        return torch.optim.Adam(model.parameters(), lr=lr)

def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on configuration"""
    scheduler_name = getattr(config, 'lr_scheduler', 'ExponentialLR')
    scheduler_params = getattr(config, 'lr_scheduler_params', {})
    
    if scheduler_name == "NoamLR":
        # Noam learning rate scheduler (for Transformer)
        warmup_steps = scheduler_params.get("warmup_steps", 4000)
        def noam_lr_lambda(step):
            if step == 0:
                step = 1
            return min(step ** (-0.5), step * warmup_steps ** (-1.5))
        scheduler = LambdaLR(optimizer, lr_lambda=noam_lr_lambda)
        print(f"✅ Created scheduler: NoamLR, Warmup steps: {warmup_steps}")
        
    elif scheduler_name == "ExponentialLR":
        # Exponential decay scheduler
        gamma = scheduler_params.get("gamma", 0.95)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        print(f"✅ Created scheduler: ExponentialLR, gamma: {gamma}")
        
    elif scheduler_name == "StepLR":
        # Step-wise scheduler
        step_size = scheduler_params.get("step_size", 1000)
        gamma = scheduler_params.get("gamma", 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"✅ Created scheduler: StepLR, step_size: {step_size}, gamma: {gamma}")
        
    else:
        # Default to exponential decay
        print(f"⚠️  Scheduler {scheduler_name} not supported, using default ExponentialLR")
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    return scheduler

class GlowTTSTrainer:
    """Simplified GlowTTS Trainer - Single GPU training only"""
    
    def __init__(
        self,
        model,
        config,
        output_path="./outputs"
    ):
        # Create directories
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")

        self.model = model.to(self.device)
        self.config = config
        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.criterion = self.model.get_criterion()
        self.output_path = output_path
        
        # Training state
        self.total_steps_done = 0
        self.epochs_done = 0
        self.best_loss = float('inf')
        
        # Data loaders
        self.train_loader = None
        self.eval_loader = None
        
        # Statistics
        self.keep_avg_train = KeepAverage()
        self.keep_avg_eval = KeepAverage()
        
    def get_train_dataloader(self) -> DataLoader:
        """Get training data loader"""
        if self.train_loader is None:
            self.train_loader = self.model.get_data_loader(
                config=self.config,
                is_eval=False
            )
        return self.train_loader
    
    def get_eval_dataloader(self) -> DataLoader:
        """Get evaluation data loader"""
        if self.eval_loader is None:
            self.eval_loader = self.model.get_data_loader(
                config=self.config,
                is_eval=True
            )
        return self.eval_loader
    
    def format_batch(self, batch: Dict) -> Dict:
        """Format batch data"""
        # formatted_batch = self.model.format_batch(batch)
        # move data to device
        for key, value in batch.items():
            batch[key] = value.to(self.device)
        return batch
    
    def train_step(self, batch: Dict) -> Tuple[Dict, Dict]:
        """Execute one training step"""
        step_start_time = time.time()

        # Directly manage data dependency initialization
        data_dep_init_steps = getattr(self.config, 'data_dep_init_steps', 0)
        self.model.run_data_dep_init = self.total_steps_done < data_dep_init_steps
        
        # Format batch data
        formatted_batch = self.format_batch(batch)
        
        # Forward pass and loss calculation
        outputs, loss_dict = self.model.train_step(formatted_batch, self.criterion)

        # Update step count
        self.total_steps_done += 1
        
        # Skip backpropagation for data dependency initialization steps
        if not loss_dict:
            return outputs, {}
        
        # Backward pass
        loss = loss_dict["loss"]
        loss.backward()
        
        # Gradient clipping
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Learning rate scheduling
        if self.scheduler is not None:
            scheduler_name = getattr(self.config, 'lr_scheduler', 'ExponentialLR')
            if scheduler_name == "NoamLR" or not getattr(self.config, 'scheduler_after_epoch', True):
                self.scheduler.step()
        
        # Calculate step time
        step_time = time.time() - step_start_time
        
        # Detach loss dictionary
        loss_dict_detached = {k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v 
                            for k, v in loss_dict.items()}
        loss_dict_detached["step_time"] = step_time
        
        return outputs, loss_dict_detached
        
    def eval_step(self, batch: Dict) -> Tuple[Dict, Dict]:
        """Execute one evaluation step"""
        with torch.no_grad():
            # Format batch data
            formatted_batch = self.format_batch(batch)
            
            # Forward pass
            outputs, loss_dict = self.model.eval_step(formatted_batch, self.criterion)
            
            if not loss_dict:
                return outputs, {}
            
            # Detach loss dictionary
            loss_dict_detached = {k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v 
                                for k, v in loss_dict.items()}
            
            return outputs, loss_dict_detached
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        torch.set_grad_enabled(True)
        
        train_loader = self.get_train_dataloader()
        epoch_start_time = time.time()
        
        print(f"🚀 Starting training Epoch {self.epochs_done}")
        
        # print(f"📊 Training data loader: {len(train_loader)} batches")
        for step, batch in enumerate(train_loader):
            # Execute training step
            outputs, loss_dict = self.train_step(batch)
            
            if not loss_dict:
                print(f" [!] Step {step} skipped (data dependency initialization)")
                continue
            
            # Update statistics
            self.keep_avg_train.update_values({f"avg_{k}": v for k, v in loss_dict.items()})
            
            # Print training progress
            if (step + 1) % self.config.print_step == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"🔄 Step {step+1}/{len(train_loader)} | "
                      f"Loss: {loss_dict['loss']:.4f} | "
                      f"Flow Loss: {loss_dict['log_mle']:.4f} | "
                      f"Duration Loss: {loss_dict['loss_dur']:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {loss_dict['step_time']:.2f}s")
            
            # Save checkpoint
            if (self.total_steps_done + 1) % self.config.save_step == 0:
                self.save_checkpoint()
        
        # Learning rate scheduling after epoch
        if self.scheduler is not None:
            scheduler_name = getattr(self.config, 'lr_scheduler', 'ExponentialLR')
            # NoamLR schedules by step, other schedulers by epoch
            if scheduler_name != "NoamLR" and getattr(self.config, 'scheduler_after_epoch', True):
                self.scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        print(f"✅ Epoch {self.epochs_done} completed, time: {epoch_time:.2f}s")
        
        # Print average statistics
        avg_stats = self.keep_avg_train.avg_values
        print(f"📊 Average Loss: {avg_stats.get('avg_loss', 0):.4f} | "
              f"Average Flow Loss: {avg_stats.get('avg_log_mle', 0):.4f} | "
              f"Average Duration Loss: {avg_stats.get('avg_loss_dur', 0):.4f}")
    
    def eval_epoch(self):
        """Evaluate one epoch"""
        self.model.eval()
        torch.set_grad_enabled(False)
        
        eval_loader = self.get_eval_dataloader()
        
        print(f"🔍 Starting evaluation Epoch {self.epochs_done}")
        
        for step, batch in enumerate(eval_loader):
            outputs, loss_dict = self.eval_step(batch)
            
            if not loss_dict:
                continue
            
            # Update statistics
            self.keep_avg_eval.update_values({f"avg_{k}": v for k, v in loss_dict.items()})
            
            # Print evaluation progress
            if (step + 1) % self.config.print_step == 0:
                print(f"📈 Evaluation step {step+1}/{len(eval_loader)} | "
                      f"Loss: {loss_dict['loss']:.4f}")
        
        # Print average statistics
        avg_stats = self.keep_avg_eval.avg_values
        avg_loss = avg_stats.get('avg_loss', 0)
        
        print(f"📊 Evaluation completed | Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_best_model()
            print(f"🎉 Found better model! Best loss: {self.best_loss:.4f}")
    
    def save_checkpoint(self):
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'total_steps_done': self.total_steps_done,
            'epochs_done': self.epochs_done,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        checkpoint_path = f"{self.output_path}/checkpoint_step_{self.total_steps_done}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_best_model(self):
        """Save best model"""
        torch.save(self.model.state_dict(), f"{self.output_path}/best_model.pth")
        print(f"🏆 Best model saved: {self.output_path}/best_model.pth")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.total_steps_done = checkpoint['total_steps_done']
        self.epochs_done = checkpoint['epochs_done']
        self.best_loss = checkpoint['best_loss']
        
        print(f"📂 Checkpoint loaded: {checkpoint_path}")
        print(f"📊 Restored state: Step {self.total_steps_done}, Epoch {self.epochs_done}, Best loss: {self.best_loss:.4f}")
    
    def fit(self):
        """Main training loop"""
        print("🎯 Starting GlowTTS training")
        print(f"📋 Configuration: {self.config.epochs} epochs, Batch size: {self.config.batch_size}")
        
        for epoch in range(self.epochs_done, self.config.epochs):
            self.epochs_done = epoch
            
            # Reset statistics
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage()
            
            # Training
            self.train_epoch()
            
            # Evaluation
            if self.config.run_eval:
                self.eval_epoch()
            
            print(f"🎊 Epoch {epoch} completed\n" + "="*50)
        
        print("🎉 Training completed!")
        print(f"🏆 Best loss: {self.best_loss:.4f}")

    def fit_from_checkpoint(self, checkpoint_path):
        """Resume training from checkpoint"""
        print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
        self.load_checkpoint(checkpoint_path)
        
        # Continue training
        self.fit()