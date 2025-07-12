import torch
import torch.nn as nn
from typing import Dict, Tuple, Union, List
import time
from coqpit import Coqpit
from utils import KeepAverage
import torch.distributed as dist
from torch.utils.data import DataLoader


def isimplemented(obj, method_name):
    """Check if a method is implemented in a class."""
    if method_name in dir(obj) and callable(getattr(obj, method_name)):
        try:
            obj.__getattribute__(method_name)()  # pylint: disable=bad-option-value, unnecessary-dunder-call
        except NotImplementedError:
            return False
        except:  # pylint: disable=bare-except
            return True
        return True
    return False


class Trainer:
    ######################
    # TRAIN FUNCTIONS
    ######################
    
    def optimize(
        self,
        batch: Dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: "AMPScaler",
        criterion: nn.Module,
        scheduler: Union[torch.optim.lr_scheduler._LRScheduler, List, Dict],  # pylint: disable=protected-access
        config: Coqpit,
        optimizer_idx: int = None,
        step_optimizer: bool = True,
        num_optimizers: int = 1,
    ) -> Tuple[Dict, Dict, int]:
        """Perform a forward - backward pass and run the optimizer.

        Args:
            batch (Dict): Input batch. If
            model (nn.Module): Model for training. Defaults to None.
            optimizer (Union[nn.optim.Optimizer, List]): Model's optimizer. If it is a list then, `optimizer_idx` must be defined to indicate the optimizer in use.
            scaler (AMPScaler): AMP scaler.
            criterion (nn.Module): Model's criterion.
            scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler used by the optimizer.
            config (Coqpit): Model config.
            optimizer_idx (int, optional): Target optimizer being used. Defaults to None.
            step_optimizer (bool, optional): Whether step the optimizer. If False, gradients are accumulated and
                model parameters are not updated. Defaults to True.
            num_optimizers (int, optional): Number of optimizers. Defaults to 1.

        Raises:
            RuntimeError: When the loss is NaN.

        Returns:
            Tuple[Dict, Dict, int, torch.Tensor]: model outputs, losses, step time and gradient norm.
        """

        step_start_time = time.time()

        # forward pass and loss computation
        outputs, loss_dict = self._compute_loss(
            batch=batch, model=model, criterion=criterion, config=config, optimizer_idx=optimizer_idx
        )

        # skip the rest if not outputs from the model
        if not loss_dict:
            step_time = time.time() - step_start_time
            return outputs, {}, step_time

        grad_clip = self._set_grad_clip_per_optimizer(config=config, optimizer_idx=optimizer_idx)
        # optimizer step
        grad_norm = 0
        update_lr_scheduler = True

        # callback
        self.callbacks.before_backward_pass(self, loss_dict)

        # accumulated gradients adjustment
        loss_dict["loss"] = loss_dict["loss"] / float(self.grad_accum_steps)

        # main model optimizer step
        loss_dict["loss"].backward()
        # gradient accumulation
        if step_optimizer:
            self.callbacks.before_gradient_clipping(self)
            if grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params(optimizer), grad_clip)
            optimizer.step()

        # setup lr
        if (
            scheduler is not None
            and update_lr_scheduler
            and not self.config.scheduler_after_epoch
            and step_optimizer
        ):
            scheduler.step()

        # zero-out optimizer
        if step_optimizer:
            optimizer.zero_grad(set_to_none=True)

        # pytorch skips the step when the norm is 0. So ignore the norm value when it is NaN
        if isinstance(grad_norm, torch.Tensor) and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            grad_norm = 0

        step_time = time.time() - step_start_time

        # detach loss dict
        loss_dict_detached = self.detach_loss_dict(loss_dict, step_optimizer, optimizer_idx, grad_norm)
        return outputs, loss_dict_detached, step_time

    def train_step(self, batch: Dict, batch_n_steps: int, step: int, loader_start_time: float) -> Tuple[Dict, Dict]:
        """Perform a training step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            batch_n_steps (int): Number of steps needed to complete an epoch. Needed for logging.
            step (int): Current step number in this epoch.
            loader_start_time (float): The time when the data loading is started. Needed for logging.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        self.callbacks.on_train_step_start(self)
        # format data
        batch = self.format_batch(batch)
        loader_time = time.time() - loader_start_time

        # conteainers to hold model outputs and losses for each optimizer.
        outputs_per_optimizer = None
        loss_dict = {}

        # OPTIMIZATION
        # gradient accumulation
        # TODO: grad accumulation for each optimizer
        step_optimizer = True
        if ((step + 1) % self.grad_accum_steps != 0) and (step + 1 != batch_n_steps):
            step_optimizer = False

        if not isinstance(self.optimizer, list):
            # auto training with a single optimizer
            outputs, loss_dict_new, step_time = self.optimize(
                batch,
                self.model,
                self.optimizer,
                self.scaler,
                self.criterion,
                self.scheduler,
                self.config,
                step_optimizer=step_optimizer,
                num_optimizers=1,
            )
            loss_dict.update(loss_dict_new)
        else:
            # auto training with multiple optimizers (e.g. GAN)
            outputs_per_optimizer = [None] * len(self.optimizer)
            total_step_time = 0
            for idx, optimizer in enumerate(self.optimizer):
                criterion = self.criterion
                # scaler = self.scaler[idx] if self.use_amp_scaler else None
                scaler = self.scaler
                scheduler = None
                if self.scheduler is not None:
                    scheduler = self.scheduler[idx]
                outputs, loss_dict_new, step_time = self.optimize(
                    batch,
                    self.model,
                    optimizer,
                    scaler,
                    criterion,
                    scheduler,
                    self.config,
                    idx,
                    step_optimizer=step_optimizer,
                    num_optimizers=len(self.optimizer),
                )
                # skip the rest if the model returns None
                total_step_time += step_time
                outputs_per_optimizer[idx] = outputs
                # merge loss_dicts from each optimizer
                # rename duplicates with the optimizer idx
                # if None, model skipped this optimizer
                if loss_dict_new is not None:
                    for k, v in loss_dict_new.items():
                        if k in loss_dict:
                            loss_dict[f"{k}-{idx}"] = v
                        else:
                            loss_dict[k] = v
                step_time = total_step_time

            outputs = outputs_per_optimizer

            # clear any pesky gradients after gradient accumulation
            if step_optimizer:
                self.model.zero_grad(set_to_none=True)

        # update avg runtime stats
        keep_avg_update = {}
        keep_avg_update["avg_loader_time"] = loader_time
        keep_avg_update["avg_step_time"] = step_time
        self.keep_avg_train.update_values(keep_avg_update)

        # update avg loss stats
        update_eval_values = {}
        for key, value in loss_dict.items():
            update_eval_values["avg_" + key] = value
        self.keep_avg_train.update_values(update_eval_values)

        # print training progress
        if self.total_steps_done % self.config.print_step == 0:
            # log learning rates
            lrs = {}
            if isinstance(self.optimizer, list):
                for idx, optimizer in enumerate(self.optimizer):
                    current_lr = self.optimizer[idx].param_groups[0]["lr"]
                    lrs.update({f"current_lr_{idx}": current_lr})
            elif isinstance(self.optimizer, dict):
                for key, optimizer in self.optimizer.items():
                    current_lr = self.optimizer[key].param_groups[0]["lr"]
                    lrs.update({f"current_lr_{key}": current_lr})
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
                lrs = {"current_lr": current_lr}

            # log run-time stats
            loss_dict.update(lrs)
            loss_dict.update(
                {
                    "step_time": round(step_time, 4),
                    "loader_time": round(loader_time, 4),
                }
            )
            self.c_logger.print_train_step(
                batch_n_steps,
                step,
                self.total_steps_done,
                loss_dict,
                self.keep_avg_train.avg_values,
            )

        if self.args.rank == 0:
            # Plot Training Iter Stats
            # reduce TB load and don't log every step
            if self.total_steps_done % self.config.plot_step == 0:
                self.dashboard_logger.train_step_stats(self.total_steps_done, loss_dict)
            if self.total_steps_done % self.config.save_step == 0 and self.total_steps_done != 0:
                if self.config.save_checkpoints:
                    # checkpoint the model
                    self.save_checkpoint()

            if self.total_steps_done % self.config.log_model_step == 0:
                # log checkpoint as artifact
                self.update_training_dashboard_logger(batch=batch, outputs=outputs)

            self.dashboard_logger.flush()

        self.total_steps_done += 1
        self.callbacks.on_train_step_end(self)
        return outputs, loss_dict

    def train_epoch(self) -> None:
        """Main entry point for the training loop. Run training on the all training samples."""
        # initialize the data loader
        if self.train_loader is None:
            self.train_loader = self.get_train_dataloader(
                self.training_assets,
                self.train_samples,
                verbose=True,
            )
            self.train_loader = self.prepare_accelerate_loader(self.train_loader)
        # set model to training mode
        torch.set_grad_enabled(True)
        if self.num_gpus > 1:
            self.model.module.train()
        else:
            self.model.train()
        epoch_start_time = time.time()

        self.callbacks.on_train_epoch_start(self)

        self.c_logger.print_train_start()
        loader_start_time = time.time()
        # TRAINING EPOCH -> iterate over the training samples
        batch_num_steps = len(self.train_loader)
        for cur_step, batch in enumerate(self.train_loader):
            outputs, _ = self.train_step(batch, batch_num_steps, cur_step, loader_start_time)
            if outputs is None:
                print(" [!] `train_step()` retuned `None` outputs. Skipping training step.")
                continue
            del outputs
            loader_start_time = time.time()

            # RUN EVAL -> run evaluation epoch in the middle of training. Useful for big datasets.
            if self.config.run_eval_steps is not None and (self.total_steps_done % self.config.run_eval_steps == 0):
                self.eval_epoch()
                if self.num_gpus > 1:
                    self.model.module.train()
                else:
                    self.model.train()
                torch.set_grad_enabled(True)

        epoch_time = time.time() - epoch_start_time
        self.callbacks.on_train_epoch_end(self)

        # scheduler step
        if self.scheduler is not None and self.config.scheduler_after_epoch:
            if isinstance(self.scheduler, list):
                for scheduler in self.scheduler:
                    if scheduler is not None:
                        scheduler.step()
            elif isinstance(self.scheduler, dict):  # only with `model.optimize()``
                for scheduler in self.scheduler.values():
                    if scheduler is not None:
                        scheduler.step()
            else:
                self.scheduler.step()
        # plot self.epochs_done Stats
        if self.args.rank == 0:
            epoch_stats = {"epoch_time": epoch_time}
            epoch_stats.update(self.keep_avg_train.avg_values)
            self.dashboard_logger.train_epoch_stats(self.total_steps_done, epoch_stats)
            if self.config.model_param_stats:
                self.dashboard_logger.model_weights(self.model, self.total_steps_done)
        torch.cuda.empty_cache()

    #######################
    # EVAL FUNCTIONS
    #######################

    def _model_eval_step(
        self, batch: Dict, model: nn.Module, criterion: nn.Module, optimizer_idx: int = None
    ) -> Tuple[Dict, Dict]:
        """
        Perform a evaluation forward pass. Compute model outputs and losses with no gradients.

        Args:
            batch (Dict): IBatch of inputs.
            model (nn.Module): Model to call evaluation.
            criterion (nn.Module): Model criterion.
            optimizer_idx (int, optional): Optimizer ID to define the closure in multi-optimizer training. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: model outputs and losses.
        """
        input_args = [batch, criterion]

        if optimizer_idx is not None:
            input_args.append(optimizer_idx)
        if hasattr(model, "module"):
            return model.module.eval_step(*input_args)

        return model.eval_step(*input_args)

    def eval_step(self, batch: Dict, step: int) -> Tuple[Dict, Dict]:
        """Perform a evaluation step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            step (int): Current step number in this epoch.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        with torch.no_grad():
            outputs = []
            loss_dict = {}
            if not isinstance(self.optimizer, list) or isimplemented(self.model, "optimize"):
                outputs, loss_dict = self._model_eval_step(batch, self.model, self.criterion)
                if outputs is None:
                    return None, None
            else:
                outputs = [None] * len(self.optimizer)
                for idx, _ in enumerate(self.optimizer):
                    criterion = self.criterion
                    outputs_, loss_dict_new = self._model_eval_step(batch, self.model, criterion, idx)
                    if outputs_ is None:
                        return None, None
                    outputs[idx] = outputs_

                    if loss_dict_new:
                        loss_dict_new[f"loss_{idx}"] = loss_dict_new.pop("loss")
                        loss_dict.update(loss_dict_new)

            loss_dict = self._detach_loss_dict(loss_dict)

            # update avg stats
            update_eval_values = {}
            for key, value in loss_dict.items():
                update_eval_values["avg_" + key] = value
            self.keep_avg_eval.update_values(update_eval_values)

            if self.config.print_eval:
                self.c_logger.print_eval_step(step, loss_dict, self.keep_avg_eval.avg_values)

        return outputs, loss_dict

    def eval_epoch(self) -> None:
        """Main entry point for the evaluation loop. Run evaluation on the all validation samples."""

        # initialize it when eval_epoch is called alone.
        self.keep_avg_eval = KeepAverage() if self.keep_avg_eval is None else self.keep_avg_eval

        if self.eval_loader is None:
            self.eval_loader = (
                self.get_eval_dataloader(
                    self.training_assets,
                    self.eval_samples,
                    verbose=True,
                )
                if self.config.run_eval
                else None
            )

        torch.set_grad_enabled(False)
        self.model.eval()
        self.c_logger.print_eval_start()
        loader_start_time = time.time()
        batch = None
        outputs = None
        for cur_step, batch in enumerate(self.eval_loader):
            # format data
            batch = self.format_batch(batch)
            loader_time = time.time() - loader_start_time
            self.keep_avg_eval.update_values({"avg_loader_time": loader_time})
            outputs_, _ = self.eval_step(batch, cur_step)
            if outputs_ is None:
                print(" [!] `eval_step()` retuned `None` outputs. Skipping evaluation step.")
                continue
            outputs = outputs_
            loader_start_time = time.time()
        # plot epoch stats, artifacts and figures
        if self.args.rank == 0 and outputs is not None:
            if hasattr(self.model, "module") and isimplemented(self.model.module, "eval_log"):
                self.model.module.eval_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            elif isimplemented(self.model, "eval_log"):
                self.model.eval_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            self.dashboard_logger.eval_stats(self.total_steps_done, self.keep_avg_eval.avg_values)
        torch.cuda.empty_cache()

    ###################################
    # FIT FUNCTIONS
    ###################################

    def _fit(self) -> None:
        """🏃 train -> evaluate -> test for the number of epochs."""
        self._restore_best_loss()

        self.total_steps_done = self.restore_step

        for epoch in range(0, self.config.epochs):
            if self.num_gpus > 1:
                # let all processes sync up before starting with a new epoch of training
                dist.barrier()
            self.callbacks.on_epoch_start(self)
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage() if self.config.run_eval else None
            self.epochs_done = epoch
            self.c_logger.print_epoch_start(epoch, self.config.epochs, self.output_path)
            if not self.skip_train_epoch and not self.start_with_eval:
                self.train_epoch()
            if self.config.run_eval:
                self.eval_epoch()
            if epoch >= self.config.test_delay_epochs and self.args.rank <= 0:
                self.test_run()

            self.c_logger.print_epoch_end(
                epoch,
                self.keep_avg_eval.avg_values if self.config.run_eval else self.keep_avg_train.avg_values,
            )
            if self.args.rank in [None, 0]:
                self.save_best_model()
            self.callbacks.on_epoch_end(self)
            self.start_with_eval = False

    def fit(self) -> None:
        """Where the ✨️magic✨️ happens..."""
        # TODO
        pass

    #########################
    # DATA LOADING FUNCTIONS
    #########################

    def get_train_dataloader(self, training_assets=None, train_samples=None, verbose=False) -> DataLoader:
        """获取训练数据加载器"""
        return self.model.get_data_loader(
            config=self.config,
            assets=training_assets,
            is_eval=False,
            samples=train_samples,
            verbose=verbose,
            num_gpus=self.num_gpus,
            rank=getattr(self.args, 'rank', 0)
        )

    def get_eval_dataloader(self, training_assets=None, eval_samples=None, verbose=False) -> DataLoader:
        """获取评估数据加载器"""
        return self.model.get_data_loader(
            config=self.config,
            assets=training_assets,
            is_eval=True,
            samples=eval_samples,
            verbose=verbose,
            num_gpus=self.num_gpus,
            rank=getattr(self.args, 'rank', 0)
        )

    def format_batch(self, batch: List) -> Dict:
        return self.model.format_batch(batch)
    