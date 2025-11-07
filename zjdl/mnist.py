import random
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import engine.lit_extension as lite
from model.model import YamlModel, Conv
from pymod.extension.path_extension import Path


def random_dropout(x, p=0.4):
    p = 1 - random.uniform(0, p)
    mask = torch.bernoulli((x > 0) * p)
    return x * mask


ckpt_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    **lite.model_ckpt_kwargs
)


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
        Conv.reparam(self.model)
        # self.to_onnx(self.project / "best.onnx", input_sample=)


CFG_MODEL = Path("config/cnn/mnist.yaml")
CFG_TRAIN = Path("config/mnist.yaml")
PROJECT = Path("runs")
DATA = Path("runs")

if __name__ == "__main__":
    module = MnistModule(YamlModel(CFG_MODEL),
                         CFG_TRAIN,
                         PROJECT,
                         ckpt_callback=ckpt_callback,
                         disable_val_prog=True)

    dl = partial(DataLoader, batch_size=module.batch_size, shuffle=True)
    dataset = MNIST(root=DATA, train=True, download=True, transform=transforms.ToTensor())
    trainset, valset = random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(0))
    trainset, valset = map(dl, (trainset, valset))

    trainer = pl.Trainer(**module.trainer_kwargs())
    trainer.fit(module, trainset, valset)
