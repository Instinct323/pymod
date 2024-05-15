import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Union


def kl_divergence(p, q, eps=1e-6):
    # KL(P||Q) = E_{P} [\log p/q]
    return (p * torch.log(p / (q + eps))).sum()


class EmaParam(nn.Parameter):

    def __new__(cls, v, momentum=1e-2):
        self = super().__new__(cls, torch.tensor(float(v)), requires_grad=False)
        self.momentum = momentum
        return self

    def update(self, v):
        self.mul_(1 - self.momentum)
        self.add_(self.momentum * v)


class FocalLoss(nn.Module):
    # :param nc (int): Number of classes

    def __init__(self, nc, gamma: float = 1.5):
        super().__init__()
        self.nc = nc
        self.gamma = gamma

    def get_target(self, target):
        return F.one_hot(target, self.nc)

    def forward(self, logits, target):
        # target: 转为 one_hot, 计算二元交叉熵
        target = self.get_target(target).float()
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        # logits: 利用 sigmoid 计算 pred, 以及聚焦系数
        if self.gamma:
            pred = logits.detach().sigmoid()
            beta = target * (1 - pred) + (1 - target) * pred
            loss *= (beta / beta.mean()) ** self.gamma
        return loss


class MultiFocalLoss(FocalLoss):
    # :param nc (tuple): Number of classes for each family
    c1 = property(lambda self: sum(1 if self.nc == 2 else x for x in self.nc))

    def get_target(self, target):
        return torch.cat([F.one_hot(target[..., i], nc) if nc > 2 else target[..., i, None]
                          for i, nc in enumerate(self.nc)], dim=-1)


class CrossEntropy(nn.Module):

    def __init__(self, softw=.6):
        super().__init__()
        assert 0 <= softw < 1
        self.softw = softw

    def forward(self, logits, hardlabel, softlabel=None):
        log_softmax = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_softmax, hardlabel)
        # softlabel
        if softlabel is not None:
            assert self.softw, "Vanishing weight"
            item = - (log_softmax * softlabel).sum(dim=-1).mean()
            item = loss * (1 - self.softw) + item * self.softw
            loss = item - item.detach() + loss.detach()
        return loss


class MultiCrossEntropy(nn.Module):
    # :param nc (tuple): Number of classes for each family
    c1 = property(lambda self: sum(self.nf))

    def __init__(self,
                 nc: Union[tuple, list],
                 w: torch.Tensor = None,
                 gamma: float = 0,
                 momentum: float = 1e-2):
        super().__init__()
        self.nf = [1 if x == 2 else x for x in nc]
        self.w = nn.Parameter(torch.ones(len(self.nf)) if w is None else w, requires_grad=False)
        self.gamma = gamma
        # 用于 FocalLoss 的概率均值
        self.pmean = EmaParam(0, momentum=momentum) if gamma else None

    def forward(self, logits, hardlabel):
        loss = torch.zeros_like(self.w)
        logits = logits.split(self.nf, dim=-1)
        for i, odd in enumerate(logits):
            # 两个类别: ce
            if odd.size(-1) == 1:
                p = odd[..., 0].sigmoid()
                t = hardlabel[..., i].float()
                nll = - (t * torch.log(p) + (1 - t) * torch.log(1 - p))
            # 大于两个类别: softmax + nll
            else:
                nll = - F.log_softmax(odd, dim=-1).gather(1, hardlabel[..., i][:, None])[:, 0]
                p = torch.exp(- nll) if self.gamma else None
            # FocalLoss
            if self.gamma:
                p = p.detach()
                self.pmean.update(p.mean())
                nll *= ((1 - p) / (1 - self.pmean)) ** self.gamma
            loss[i] = nll.mean()
        return (loss * self.w).sum()

    def predict(self, logits):
        logits = logits.split(self.nf, dim=-1)
        # 两个类别: 根据 sigmoid 函数可知, logits > 0 等价于 p > .5
        return torch.stack([(odd[..., 0] > 0).long() if odd.size(-1) == 1
                            else odd.argmax(-1) for odd in logits], dim=-1)


class ContrastiveLoss(nn.Module):
    """ 同数据类型的对比损失
        g: 增强数据相对于原数据的倍数"""
    t = property(lambda self: self.param_t.sigmoid() * 4.5 + .5)

    def __init__(self, g=1, dim=-1, const_t=False):
        super().__init__()
        assert g > 0, f"Invalid value g={g}"
        self.g = g
        self.param_t = nn.Parameter(torch.tensor(-2.079), requires_grad=not const_t)
        self.measure = nn.CosineSimilarity(dim=dim, eps=1e-6)

    def extra_repr(self):
        return f"g={self.g}, \n" \
               f"T={self.t.item()}"

    def gs(self, x):
        B, C = x.shape
        assert B % (self.g + 1) == 0, "batch_size is not match g"
        return B // (self.g + 1)

    def forward(self, x):
        gs = self.gs(x)
        measure = (self.measure(x, x[:, None]) / self.t).exp() * (1 - torch.eye(int(x.size(0)))).to(x)
        # 创建掩膜, 分离正负对的数据
        mask = torch.eye(gs).repeat(self.g + 1, self.g + 1).to(x)
        p = (mask * measure).sum(dim=-1)
        n = ((1 - mask) * measure).sum(dim=-1)
        return - torch.log(p / (n + p.detach())).mean()

    def accuracy(self, x):
        """ :return: 正样本数, 总样本数"""
        B = int(x.size(0))
        gs = self.gs(x)
        measure = self.measure(x, x[:, None]) * (1 - torch.eye(B)).to(x)
        # 每个样本匹配 g 个样本
        topk = torch.argsort(measure, dim=-1, descending=True)[:, :self.g].to(x)
        match = (topk % gs == (torch.arange(B)[:, None].to(x) % gs)).sum().item()
        return np.array([match, topk.numel()])


if __name__ == "__main__":
    logits = torch.rand([9, 5], requires_grad=True)
    target = torch.randint(0, 2, [9, 3])

    fl = MultiCrossEntropy((3, 2, 2), momentum=1.)
    print(logits)
    print(fl(logits, target))
    print(fl.pmean)
    print(fl.predict(logits))
