import torch as th
import copy
from tqdm import tqdm
from resample import UniformSampler
from nn import update_ema
from torch.optim import AdamW
import time
import os


class Trainer:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        log_interval,
        save_interval,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        save_dir="",
        mmd_alpha=0.0005,
    ):

        # Initialize the model, data, and optimizer
        self.data = data
        self.batch_size = batch_size
        self.diffusion = diffusion
        self.model = model
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.cuda = th.cuda.is_available()
        self.device = th.device("cuda") if self.cuda else th.device("cpu")
        if self.cuda:
            self.model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Initialize the training loop parameters
        self.step = 0
        self.lr_anneal_steps = lr_anneal_steps

        # Log and save intervals
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.alpha = mmd_alpha

    def train(self):
        """
        Train the model for a given number of steps.
        """
        self.step = 0

        with tqdm(total=self.lr_anneal_steps) as pbar:
            while self.step < self.lr_anneal_steps:
                self.opt.zero_grad()
                data = next(self.next_batch()).to(self.device)
                t, weights = self.schedule_sampler.sample(self.batch_size, self.device)
                losses = self.diffusion.training_losses(self.model, data, t)
                loss = (losses["mse"] * weights).mean()
                loss_str = f"mse: {loss:.6f}"
                if "mmd" in losses:
                    mmd_loss = self.alpha * losses["mmd"]
                    loss += mmd_loss
                    loss_str += f", mmd: {mmd_loss:.6f}"
                loss_str += f", total: {loss:.6f}"
                pbar.set_description(loss_str)
                loss.backward()
                self._anneal_lr()
                self.opt.step()
                self.step += 1
                if self.step % self.save_interval == 0:
                    self.save()
                pbar.update(1)
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def next_batch(self):
        """
        Get the next batch of data.
        """
        while True:
            for data in self.data:
                yield data

    def save(self):
        def save_checkpoint(rate, params):
            if not rate:
                state_dict = self.model.state_dict()
                filename = f"model_{(self.step):06d}.pt"
            else:
                state_dict = self._ema_params_to_state_dict(params)
                filename = f"ema_{rate}_{(self.step):06d}.pt"
            th.save(
                {
                    "step": self.step,
                    "model_state_dict": state_dict,
                    "opt_state_dict": self.opt.state_dict(),
                },
                f"{self.save_dir}{filename}",
            )

        save_checkpoint(0, self.model)
