import random

import numpy as np
import torch
import torch.nn.functional as F
from model.common import LOCAL

from .crosstab import Crosstab
from .loss import ContrastiveLoss
from .trainer import Trainer


def linear_probing(train_x, train_y, eval_x=None, eval_y=None,
                   hlayer=tuple(), **mlp_kwd) -> Crosstab:
    # x: 标准化后的数据
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=hlayer, **mlp_kwd)
    mlp.fit(train_x, train_y)
    return Crosstab(mlp.predict(eval_x), eval_y) \
        if isinstance(eval_y, np.ndarray) else Crosstab(mlp.predict(train_x), train_y)


class Grader(Trainer):
    ''' 分级训练器'''


class ContrastiveLearn(Trainer):
    ''' 对比学习
        tf: 数据增强时使用的变换器'''

    def __init__(self, model, project, m_title, hyp, tf):
        # 获取数据增强模块
        self._tf = lambda x: torch.stack([tf(i) for i in x])
        # 将对比损失的温度参数写入模型参数
        self.cl = ContrastiveLoss(g=1)
        model.t = self.cl.param_t
        super().__init__(model, project, m_title, hyp)

    def forward(self, x):
        x = torch.cat((x, self._tf(x)))
        return self.model(x)

    def loss(self, image, target):
        pred = self.forward(image)
        return self.cl(pred)

    def metrics(self, generator) -> np.ndarray:
        for batch in generator: pass
        raise NotImplementedError

    def fitness(self, metrics) -> float:
        raise NotImplementedError


class MaskedAutoEncoder(Trainer):
    ''' 仅适用于经典 ViT
        npatch: patches 的总数
        pmask: patches 的固定掩码
        drop: patches 的遮蔽比例
        downsample: 复原图像的下采样比例'''

    def __init__(self, model, project, m_title, hyp,
                 npatch, pmask=None, drop=.75, downsample=4):
        self.pmask = pmask if pmask is not None else torch.ones(npatch, dtype=torch.bool)
        self.pperm = torch.nonzero(self.pmask).flatten().tolist()
        self.select = round(len(self.pperm) * (1 - drop))
        # 下采样函数
        self.downsample = lambda x: x[..., downsample // 2::downsample, downsample // 2::downsample]
        super().__init__(model, project, m_title, hyp)

    def loss(self, image, target) -> torch.Tensor:
        random.shuffle(self.pperm)
        select = torch.tensor(self.pperm[:self.select])
        # pmask: 有效的、未遮挡的区域
        LOCAL.pmask = torch.index_fill(torch.zeros_like(self.pmask), dim=0, index=select, value=True)
        # mmask: 有效的、被遮挡的区域
        mmask = (LOCAL.pmask ^ self.pmask).to(image)
        # pred[B, L, C]
        pred = self.model(image)
        # img[B, C, H, W] -> img[B, L, C]
        image = self.downsample(image).flatten(start_dim=2).transpose(1, 2)
        return F.mse_loss(pred[:, mmask], image[:, mmask])

    def metrics(self, generator) -> np.ndarray:
        for batch in generator: pass
        raise NotImplementedError

    def fitness(self, metrics) -> float:
        raise NotImplementedError


if __name__ == '__main__':
    x = np.random.random([100, 5])
    y = np.random.randint(0, 3, 100)
    print(linear_probing(x, y))
