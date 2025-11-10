import random

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn

import utils.lit_extension as lite
from model.common import Conv2d, CspOSA
from pymod.extension.path_extension import Path

torch.set_float32_matmul_precision("medium")
pl.seed_everything(seed=0, workers=True)

CFG_TRAIN = Path("config/mnist/hyp.yaml")
PROJECT = Path("runs")
DATA = Path("runs")


def random_dropout(x, p=0.4):
    p = 1 - random.uniform(0, p)
    mask = torch.bernoulli((x > 0) * p)
    return x * mask


ckpt_callback = pl.callbacks.ModelCheckpoint(
    **lite.model_ckpt_kwargs,
    monitor="val_acc",
    mode="max",
)


class SimpleCNN(nn.ModuleList):

    def __init__(self):
        super().__init__()
        self.append(Conv2d(1, 8, k=3, s=2))
        self.append(Conv2d(self[-1].c2, 16, k=3))
        self.append(nn.MaxPool2d(2, 2))
        self.append(CspOSA(self[-2].c2, 16, e=3, n=3))
        self.append(nn.AdaptiveAvgPool2d(1))
        self.append(nn.Conv2d(self[-2].c2, 10, kernel_size=1))
        self.append(nn.Flatten())

    def forward(self, x):
        for m in self: x = m(x)
        return x


class MnistModule(lite.LitModule):

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(random_dropout(x))
        return F.cross_entropy(pred, y) + 0.05 * self.ema_mse(x, pred)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)

        self.log_dict({
            "count": len(y),
            "correct": (pred.argmax(dim=1) == y).sum().item()
        }, on_epoch=True, reduce_fx=torch.sum)
        return F.cross_entropy(pred, y)

    def on_validation_epoch_end(self):
        print()
        metrics = self.trainer.callback_metrics
        count, correct = metrics["count"].item(), metrics["correct"].item()
        acc = correct / count if count > 0 else 0.

        self.log("val_acc", acc, prog_bar=True)
        print(f"Accuracy {acc:.4f}")

    def on_fit_end(self):
        self.load_best_ckpt()
        Conv2d.reparam(self.model)
        # self.to_onnx(self.project / "best.onnx", input_sample=)


if __name__ == "__main__":
    dataset = MNIST(root=DATA, train=True, download=True, transform=transforms.ToTensor())

    module = MnistModule(SimpleCNN(),
                         CFG_TRAIN,
                         PROJECT,
                         ckpt_callback=ckpt_callback,
                         disable_val_prog=True)

    datamodule = pl.LightningDataModule.from_datasets(
        *random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(0)),
        batch_size=module.batch_size
    )

    trainer = pl.Trainer(**module.trainer_kwargs())
    trainer.fit(module, datamodule=datamodule)
