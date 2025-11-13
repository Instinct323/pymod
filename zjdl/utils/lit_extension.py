import math
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from tqdm import tqdm


class TrainOnlyProgressBar(pl.callbacks.TQDMProgressBar):

    def init_validation_tqdm(self):
        return tqdm(disable=True)


class CosineLR(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, lrf, epochs):
        lr_lambda = lambda x: lrf + (1 + math.cos(math.pi * x / epochs)) / 2 * (1 - lrf)
        super().__init__(optimizer, lr_lambda)


class LitTopModule(pl.LightningModule):

    def __init__(self,
                 config: dict):
        super().__init__()
        self.ema: EMA = None
        self.config: dict = config

    def configure_model(self):
        """ Configure models. """
        if not self.ema and self.config.get("ema"):
            self.ema = EMA(self, **self.config["ema"], include_online_model=False)

    def configure_optimizers(self):
        """ Configure optimizers and learning rate schedulers. """
        config = self.config["optimizer"]
        optimizer = getattr(torch.optim, config["type"])(
            self.parameters(), lr=config["lr0"], **config["kwargs"]
        )
        lr_scheduler = CosineLR(optimizer, lrf=config["lrf"], epochs=self.trainer.max_epochs)
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

    def on_after_backward(self):
        """ Update EMA model after each backward pass. """
        if self.ema: self.ema.update()

    def trainer_kwargs(self,
                       ckpt_callback: pl.callbacks.ModelCheckpoint,
                       disable_val_prog: bool = False) -> dict:
        """ Get pl.Trainer keyword arguments. """
        callbacks = [ckpt_callback]
        if disable_val_prog: callbacks.append(TrainOnlyProgressBar())

        output: Path = Path(self.config["output"])
        print(f"To start tensorboard, run: `tensorboard --logdir={output.resolve()}`")

        return dict(
            max_epochs=self.config["epochs"],
            default_root_dir=output, callbacks=callbacks,
            enable_checkpointing=True, enable_progress_bar=True, enable_model_summary=True
        )
