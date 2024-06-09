import random
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .crosstab import Crosstab
from .trainer import Trainer
from ..model.common import LOCAL


def linear_probing(x, y, hlayer=tuple(), cv=5, seed=0, **mlp_kwd) -> Crosstab:
    from sklearn.model_selection import cross_val_predict
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    np.random.seed(seed)
    # 对 x 进行标准化
    x = StandardScaler().fit_transform(x)
    mlp = MLPClassifier(hidden_layer_sizes=hlayer, **mlp_kwd)
    return Crosstab(cross_val_predict(mlp, x, y, cv=cv), y)


class SimSiam(Trainer):

    def __init__(self, model, project, hyp,
                 head: Union[nn.Conv2d, nn.Linear]):
        self.head = model.head = head
        super().__init__(model, project, hyp)

    def loss(self, origin, aug) -> torch.Tensor:
        proj = self.model(torch.cat([origin, aug], dim=0))
        pred = self.head(proj).chunk(2, dim=0)
        proj = proj.detach().chunk(2, dim=0)
        return - F.cosine_similarity(pred[0], proj[1], dim=-1, eps=1e-6).mean() \
            - F.cosine_similarity(pred[1], proj[0], dim=-1, eps=1e-6).mean()


class MaskedAutoEncoder(Trainer):
    """ 仅适用于经典 ViT
        :param npatch: patches 的总数
        :param pmask: patches 的固定掩码
        :param drop: patches 的遮蔽比例
        :param downsample: 复原图像的下采样比例"""

    def __init__(self, model, project, hyp,
                 npatch, pmask=None, drop=.75, downsample=4):
        self.pmask = pmask if pmask is not None else torch.ones(npatch, dtype=torch.bool)
        self.pperm = torch.nonzero(self.pmask).flatten().tolist()
        self.select = round(len(self.pperm) * (1 - drop))
        # 下采样函数
        self.downsample = lambda x: x[..., downsample // 2::downsample, downsample // 2::downsample]
        super().__init__(model, project, hyp)

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


if __name__ == "__main__":
    x = np.random.random([100, 5])
    y = np.random.randint(0, 3, 100)
    print(linear_probing(x, y))
