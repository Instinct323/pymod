import shutil
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from ema_pytorch import EMA
from torch import nn


def wait_for_cuda(processes: list[str],
                  sleep: float = 10.):
    import nvitop, time

    devices = nvitop.Device.all()
    while True:
        wait = False
        for dev in devices:
            for process in dev.processes().values():
                wait |= process.host.name() in processes
        if not wait: break
        print(f"Waiting for CUDA devices to be free...")
        time.sleep(sleep)


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

    def copy_into_project(self, file: Path):
        file = Path(file)
        shutil.copy(file, self.project / file.name)

    @property
    def ema_forward(self):
        if self.ema is None:
            warnings.warn("no EMA model found.")
        elif self.ema.initted.item():
            return self.ema.forward_eval

    def ema_mse(self, x, y) -> torch.Tensor:
        """ Calculate EMA MSE loss. """
        func = self.ema_forward
        if not func: return 0.

        loss = F.mse_loss(func(x), y)
        return loss - loss.detach()

    def on_fit_start(self):
        self.copy_into_project(self.config_file)

    def on_after_backward(self):
        """ Update EMA model after each backward pass. """
        if self.ema: self.ema.update()

    @property
    def project(self) -> Path:
        return Path(self.trainer.log_dir).resolve()

    def trainer_kwargs(self,
                       callbacks: list[pl.callbacks.Callback] = None) -> dict:
        """ Get pl.Trainer keyword arguments. """
        callbacks = callbacks or []

        config_metric = self.config.get("metric")
        if config_metric:
            monitor_kwargs = dict(monitor=config_metric["monitor"], mode=config_metric["mode"])
            callbacks.append(pl.callbacks.ModelCheckpoint(**monitor_kwargs, filename="best", save_last=True))

            if config_metric.get("patience"):
                callbacks.append(pl.callbacks.EarlyStopping(**monitor_kwargs, patience=config_metric["patience"], verbose=True))

        output: Path = Path(self.config["train"]["output"])
        print(f"To start tensorboard, run: `tensorboard --logdir={output.resolve()}`")

        return dict(
            max_epochs=self.config["train"]["epochs"],
            accumulate_grad_batches=self.config["train"].get("accum_grad", 1),
            default_root_dir=output, callbacks=callbacks,
            enable_checkpointing=True, enable_progress_bar=True, enable_model_summary=True
        )

    def _shared_step(self, stage, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self._shared_step("train", batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self._shared_step("val", batch, batch_idx)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self._shared_step("test", batch, batch_idx)
