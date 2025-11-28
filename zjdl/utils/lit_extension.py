import shutil
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from ema_pytorch import EMA
from torch import nn
from tqdm import tqdm


class TrainOnlyProgressBar(pl.callbacks.TQDMProgressBar):

    def init_validation_tqdm(self):
        return tqdm(disable=True)


class LitTopModule(pl.LightningModule):

    def __init__(self,
                 config_file: Path,
                 ema_model: nn.Module = None):
        super().__init__()
        self.ema: EMA = ema_model
        self.config_file: Path = config_file.resolve()
        self.config: dict = yaml.load(self.config_file.read_text(), Loader=yaml.Loader)
        pl.seed_everything(seed=self.config["train"].get("seed"), workers=True)

    def configure_model(self):
        """ Configure models. """
        ema_param = self.config["train"].get("ema")
        if not isinstance(self.ema, EMA) and ema_param:
            self.ema = EMA(self.ema or self, **ema_param, include_online_model=False)

    def configure_optimizers(self):
        """ Configure optimizers and learning rate schedulers. """
        config = self.config["train"]["optimizer"]
        optimizer = getattr(torch.optim, config["type"])(
            self.parameters(), lr=config["lr0"], **config["kwargs"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=config["lr0"] * config["lrf"],
                last_epoch=self.trainer.current_epoch - 1
            ), "interval": "epoch"}
        }

    def ema_mse(self, x, y) -> torch.Tensor:
        """ Calculate EMA MSE loss. """
        if self.ema is None:
            warnings.warn("no EMA model found.")
            return 0

        loss = F.mse_loss(self.ema.forward_eval(x), y)
        return loss - loss.detach()

    def on_fit_start(self):
        dst = Path(self.trainer.log_dir).resolve() / "module.yaml"
        shutil.copy(self.config_file, dst)

    def on_after_backward(self):
        """ Update EMA model after each backward pass. """
        if self.ema: self.ema.update()

    def trainer_kwargs(self,
                       ckpt_callback: pl.callbacks.ModelCheckpoint,
                       disable_val_prog: bool = False) -> dict:
        """ Get pl.Trainer keyword arguments. """
        callbacks = [ckpt_callback]
        if disable_val_prog: callbacks.append(TrainOnlyProgressBar())

        output: Path = Path(self.config["train"]["output"])
        print(f"To start tensorboard, run: `tensorboard --logdir={output.resolve()}`")

        return dict(
            max_epochs=self.config["train"]["epochs"],
            accumulate_grad_batches=self.config["train"].get("accum_grad", 1),
            default_root_dir=output, callbacks=callbacks,
            enable_checkpointing=True, enable_progress_bar=True, enable_model_summary=True
        )
