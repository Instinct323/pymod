import random
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from engine.result import Result
from engine.trainer import Trainer
from model.ema import EmaModel
from model.model import YamlModel, Conv
from utils.utils import *


def random_dropout(x, p=0.4):
    p = 1 - random.uniform(0, p)
    mask = torch.bernoulli((x > 0) * p)
    return x * mask


CFG = Path("cfg/cnn/mnist.yaml")
HYP = Path("cfg/hyp.yaml")
PROJECT = Path("runs")
DATA = Path("data")

BATCH_SIZE = 1000

if __name__ == "__main__":
    model_orig = YamlModel(CFG)
    model_call = model_orig.DP()
    ema_mse = EmaModel(model_orig, bp_times=50).DP().mse

    # torch 2.0.0 新特性
    if int(torch.__version__[0]) >= 2:
        model_call, ema_mse = map(torch.compile, (model_call, ema_mse))

    # 读取数据集
    dl = partial(DataLoader, batch_size=BATCH_SIZE, shuffle=True)
    dataset = MNIST(root=DATA, train=True, download=True, transform=transforms.ToTensor())
    trainset, valset = random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(0))
    trainset, valset = map(dl, (trainset, valset))

    # 启动训练器
    trainer = Trainer(model_orig, PROJECT, HYP)
    best = PROJECT / "best.pt"
    fitness = torch.load(best)["fitness"] if best.exists() else 0
    print("fitness:", fitness)

    # 记录器
    result = Result(PROJECT, ("acc",))

    for epoch in trainer:
        # 训练模型
        model_orig.train()
        qbar = tqdm(trainset)
        for img, tar in qbar:
            img = img.cuda()
            pred = model_call(random_dropout(img))
            loss = F.cross_entropy(pred, tar.cuda()) + ema_mse(img, pred)
            trainer.bp_gradient(loss)
            qbar.set_description(f"Epoch {epoch}, loss {loss.item()}")
        trainer.save_ckpt()

        # 验证模型
        with torch.no_grad():
            model_orig.eval()
            cnt, correct = 0, 0

            for img, tar in valset:
                pred = model_call(img.cuda()).argmax(axis=1).cpu()
                cnt += len(pred)
                correct += (pred == tar).sum().item()
            acc = correct / cnt
            result.record((acc,))
            print(f"Epoch {epoch}, Accuracy {acc:.4f}")

            save_list = ["last.pt"]
            if acc > fitness:
                save_list.append("best.pt")
                fitness = acc
            trainer.save_ckpt(save_list, fitness=acc)

    # 保存模型
    Conv.reparam(model_orig)
    model_orig.onnx(PROJECT / "mnist.onnx", branch="ema")
