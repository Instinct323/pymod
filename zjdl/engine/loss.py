import torch
import torch.nn.functional as F
from torch import nn


def kl_divergence(p, q, eps=1e-6):
    # KL(P||Q) = E_{P} [\log p/q]
    return (p * torch.log(p / (q + eps))).sum()


class FocalLoss(nn.Module):

    def __init__(self, nc, gamma: float = 1.5):
        super().__init__()
        self.nc = nc
        self.gamma = gamma

    def get_target(self, target):
        return F.one_hot(target, self.nc)

    def forward(self, logits, target):
        # target: 转为 one_hot, 计算二元交叉熵
        target = self.get_target(target).float()
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        # logits: 利用 sigmoid 计算 pred, 以及聚焦系数
        if self.gamma:
            pred = logits.detach().sigmoid()
            beta = target * (1 - pred) + (1 - target) * pred
            loss *= (beta / beta.mean()) ** self.gamma
        return loss


class MultiFocalLoss(FocalLoss):

    def get_target(self, target):
        for x in self.nc:
            pass


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
            assert self.softw, 'Vanishing weight'
            item = - (log_softmax * softlabel).sum(dim=-1).mean()
            item = loss * (1 - self.softw) + item * self.softw
            loss = item - item.detach() + loss.detach()
        return loss


class ContrastiveLoss(nn.Module):
    ''' 同数据类型的对比损失
        g: 增强数据相对于原数据的倍数'''
    t = property(fget=lambda self: self.param_t.sigmoid() * 4.5 + .5)

    def __init__(self, g=1, dim=-1):
        super().__init__()
        assert g > 0, f'Invalid value g={g}'
        self.g = g
        self.param_t = nn.Parameter(torch.tensor(1.))
        self.measure = nn.CosineSimilarity(dim=dim, eps=1e-6)

    def extra_repr(self):
        return f'group={self.g}, \n' \
               f'overlap={self.overlap}, \n' \
               f'T={self.t.item()}'

    def gs(self, x):
        B, C = x.shape
        assert B % (self.g + 1) == 0, 'batch_size is not match g'
        return B // (self.g + 1)

    def forward(self, x):
        gs = self.gs(x)
        measure = (self.measure(x, x[gs:, None]) / self.t).exp()
        # 分离出正对的数据
        posi = torch.cat([torch.diag(measure.narrow(0, i * gs, gs)[:, :gs]) for i in range(self.g)])
        measure = measure * (1 - torch.eye(gs).repeat(self.g, self.g + 1).to(x.device))
        return - torch.log(posi / (measure.sum(dim=-1) + posi.detach())).mean()

    def accuracy(self, x):
        gs = self.gs(x)
        measure = self.measure(x, x[gs:, None]).exp()
        # 分离出正对的数据
        posi = torch.cat([torch.diag(measure.narrow(0, i * gs, gs)[:, :gs]) for i in range(self.g)])
        measure = measure * (1 - torch.eye(gs).repeat(self.g, self.g + 1).to(x.device))
        return (posi > measure.max(dim=-1)[0]).float().mean().item()


if __name__ == '__main__':
    logits = torch.rand([9, 4], requires_grad=True)
    target = torch.randint(0, 4, [9])

    for g in (0, 1):
        fl = CrossEntropy(l2penalty=g)

        loss = fl(logits, target)
        print(loss)
    # print(F.log_softmax(logits))
    # print(F.cross_entropy(logits, target, reduction='none'))
