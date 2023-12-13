from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from engine.trainer import TrainerBase
from model.model import YamlModel, Conv
from utils.utils import *

CFG = None
HYP = None
PROJECT = Path('runs')
DATA = Path('data')

BATCH_SIZE = 1000

if __name__ == '__main__':
    m = YamlModel(CFG)
    trainer = TrainerBase(m, PROJECT, HYP)

    # 读取数据集
    dl = partial(DataLoader, batch_size=BATCH_SIZE, shuffle=True)
    dataset = MNIST(root=DATA / 'train', train=True, download=True, transform=transforms.ToTensor())
    trainset, valset = map(dl, random_split(dataset, [50000, 10000]))

    for epoch in trainer:
        # 训练模型
        m.train()
        qbar = tqdm(trainset)
        for img, tar in qbar:
            pred = m(img.cuda())
            loss = F.cross_entropy(pred, tar.cuda())
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
                correct += (pred == tar).sum()
            print(f'Epoch {epoch}, Accuracy {correct / cnt}')

    # 保存模型
    Conv.reparam(m)
    m.onnx('vovnet.onnx')
