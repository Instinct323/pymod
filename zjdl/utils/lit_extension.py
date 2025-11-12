import math
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from ema_pytorch import EMA
from torch import nn
from tqdm import tqdm

model_ckpt_kwargs = dict(filename="best", save_last=True)


class TrainOnlyProgressBar(pl.callbacks.TQDMProgressBar):

    def init_validation_tqdm(self):
        return tqdm(disable=True)


class CosineLR(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, lrf, epochs):
        lr_lambda = lambda x: lrf + (1 + math.cos(math.pi * x / epochs)) / 2 * (1 - lrf)
        super().__init__(optimizer, lr_lambda)


class LitModule(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 cfg: dict | Path,
                 ckpt_callback: pl.callbacks.ModelCheckpoint,
                 disable_val_prog: bool = False):
        super().__init__()
        assert ckpt_callback.save_last, "ckpt_callback must save last checkpoint."

        self.model: nn.Module = model.train()
        self.ema: EMA = None
        self.ckpt_callback: pl.callbacks.ModelCheckpoint = ckpt_callback

        self.cfg: dict = cfg if isinstance(cfg, dict) \
            else yaml.load(cfg.read_text(), Loader=yaml.Loader)
        self.output: Path = Path(self.cfg["output"])
        print(f"To start tensorboard, run: `tensorboard --logdir={self.output.resolve()}`")

        self.batch_size = self.cfg["batch_size"]
        self.disable_val_prog: bool = disable_val_prog

    def configure_model(self):
        """ Configure models. """
        if not self.ema and self.cfg.get("ema"):
            self.ema = EMA(self.model, **self.cfg["ema"], include_online_model=False)

    def configure_optimizers(self):
        """ Configure optimizers and learning rate schedulers. """
        cfg = self.cfg["optimizer"]
        optimizer = getattr(torch.optim, cfg["type"])(
            self.model.parameters(), lr=cfg["lr0"], weight_decay=cfg["weight_decay"]
        )
        lr_scheduler = CosineLR(optimizer, lrf=cfg["lrf"], epochs=self.trainer.max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}
        }

    def ema_mse(self, x, y) -> torch.Tensor:
        """ Calculate EMA MSE loss. """
        if self.ema is None:
            warnings.warn("no EMA model found.")
            return 0

        loss = F.mse_loss(self.ema.forward_eval(x), y)
        return loss - loss.detach()

    def load_best_ckpt(self):
        """ Load best checkpoint. """
        p = self.ckpt_callback.best_model_path
        assert p, "no best model found."
        self.load_state_dict(torch.load(p, weights_only=True)["state_dict"])

    def on_after_backward(self):
        """ Update EMA model after each backward pass. """
        if self.ema: self.ema.update()

    def trainer_kwargs(self) -> dict:
        """ Get pl.Trainer keyword arguments. """
        callbacks = []
        if self.ckpt_callback: callbacks.append(self.ckpt_callback)
        if self.disable_val_prog: callbacks.append(TrainOnlyProgressBar())

        return dict(
            max_epochs=self.cfg["epochs"],
            default_root_dir=self.output, callbacks=callbacks,
            enable_checkpointing=True, enable_progress_bar=True, enable_model_summary=True
        )
