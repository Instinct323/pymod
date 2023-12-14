import random
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from engine.trainer import Trainer
from model.ema import EmaModel
from model.model import YamlModel, Conv
from utils.utils import *


def random_dropout(x, p=0.4):
    p = 1 - random.uniform(0, p)
    mask = torch.bernoulli((x > 0) * p)
    return x * mask


CFG = Path('config/cnn/mnist.yaml')
HYP = Path('config/hyp.yaml')
PROJECT = Path('runs')
DATA = Path('data')

BATCH_SIZE = 1000

if __name__ == '__main__':
    # 读取数据集
    dl = partial(DataLoader, batch_size=BATCH_SIZE, shuffle=True)
    dataset = MNIST(root=DATA, train=True, download=True, transform=transforms.ToTensor())
    trainset, valset = random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(0))

    # 加载模型
    m = YamlModel(CFG)
    ema = EmaModel(m, bp_times=50)
    trainset, valset = map(dl, (trainset, valset))

    # 启动训练器
    trainer = Trainer(m, PROJECT, HYP)
    best = PROJECT / 'best.pt'
    fitness = torch.load(best)['fitness'] if best.exists() else 0
    print('fitness:', fitness)

    for epoch in trainer:
        # 训练模型
        m.train()
        qbar = tqdm(trainset)
        for img, tar in qbar:
            img = img.cuda()
            pred = m(random_dropout(img))
            loss = F.cross_entropy(pred, tar.cuda()) + ema.mse(img, pred)
            trainer.bp_gradient(loss)
            qbar.set_description(f'Epoch {epoch}, loss {loss.item()}')
        trainer.save_ckpt()

        # 验证模型
        with torch.no_grad():
            m.eval()
            cnt, correct = 0, 0

            for img, tar in valset:
                pred = m(img.cuda()).argmax(axis=1).cpu()
                cnt += len(pred)
                correct += (pred == tar).sum().item()
            acc = correct / cnt
            print(f'Epoch {epoch}, Accuracy {acc:.4f}')

            save_list = ['last.pt']
            if acc > fitness:
                save_list.append('best.pt')
                fitness = acc
            trainer.save_ckpt(save_list, fitness=acc)

    # 保存模型
    Conv.reparam(m)
    m.onnx('mnist.onnx')
