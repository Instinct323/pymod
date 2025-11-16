import random

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import utils.lit_extension as lite
from model import common
from pymod.extension.path_extension import Path

torch.set_float32_matmul_precision("medium")
pl.seed_everything(seed=0, workers=True)

CFG_TRAIN = Path("config/mnist/hyp.yaml")
DATA = Path("runs")


def random_dropout(x, p=0.4):
    p = 1 - random.uniform(0, p)
    mask = torch.bernoulli((x > 0) * p)
    return x * mask


ckpt_callback = pl.callbacks.ModelCheckpoint(
    filename="best", save_last=True,
    monitor="acc_val", mode="max",
)


class MnistModule(lite.LitTopModule):

    def __init__(self):
        super().__init__(CFG_TRAIN)
        self.model = nn.ModuleList()
        self.model.append(common.ConvBnAct2d(1, 8, k=3, s=2))
        self.model.append(common.ConvBnAct2d(self.model[-1].c2, 16, k=3))
        self.model.append(nn.MaxPool2d(2, 2))
        self.model.append(common.CspOSA(self.model[-2].c2, 16, e=3, n=3))
        self.model.append(nn.AdaptiveAvgPool2d(1))
        self.model.append(nn.Conv2d(self.model[-2].c2, 10, kernel_size=1))
        self.model.append(nn.Flatten())

    def forward(self, x):
        for m in self.model: x = m(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(random_dropout(x))
        loss = F.cross_entropy(pred, y)
        self.log("loss_train", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss + 0.05 * self.ema_mse(x, pred)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)

        self.log("loss_val", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("acc_val", (pred.argmax(dim=1) == y).float().mean(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        print()
        metrics = self.trainer.callback_metrics
        acc = metrics["val_acc"]
        print(f"Accuracy {acc:.4f}")

    def on_fit_end(self):
        self.eval()
        common.fuse_modules(self)
        # self.to_onnx(self.project / "best.onnx", input_sample=)


if __name__ == "__main__":
    dataset = MNIST(root=DATA, train=True, download=True, transform=transforms.ToTensor())

    module = MnistModule()

    datamodule = pl.LightningDataModule.from_datasets(
        *random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(0)),
        batch_size=module.config["train"]["batch_size"]
    )

    trainer = pl.Trainer(**module.trainer_kwargs(ckpt_callback=ckpt_callback, disable_val_prog=True))
    trainer.fit(module, datamodule=datamodule)
