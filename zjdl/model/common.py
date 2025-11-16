from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sum_ = lambda x: sum(x[1:], x[0])
make_divisible = lambda x, divisor=4: np.maximum(np.round(x / divisor).astype(np.int64), 1) * divisor


def apply_width_multiplier(channels: list,
                           w: float,
                           divisor: int = 4) -> list:
    """ Apply width multiplier to channels. """
    return [
        apply_width_multiplier(c, w, divisor) if isinstance(c, list) else make_divisible(c * w, divisor=4)
        for c in channels
    ]


def fuse_modules(model: nn.Module):
    for m in filter(lambda m: isinstance(m, _ConvBnActNd) and not m.deploy, model.modules()):
        torch.ao.quantization.fuse_modules(m, ["conv", "bn"], inplace=True)


class _ConvBnActNd(nn.Module):
    """ Conv - BN - Act """
    deploy = property(lambda self: isinstance(self.bn, nn.Identity))
    ConvType = None
    BnType = None

    @staticmethod
    def auto_pad(k, s=1, d=1):
        # (k - 1) // 2 * d: 1st-center -> [0, 0]
        # (s - 1) // 2: 1st-center -> [s/2, s/2]
        return max(0, (k - 1) // 2 * d - (s - 1) // 2)

    def __init__(self, c1, c2, k=3, s=1, g=1, d=1,
                 act: Optional[nn.Module] = nn.ReLU, ctrpad=True):
        super().__init__()
        self.c2 = c2
        assert k & 1, "The convolution kernel size must be odd"
        # depthwise separable convolution
        if g == "dw":
            g = c1
            assert c1 == c2, "Failed to create DWConv"
        padding = self.auto_pad(k, s if ctrpad else 1, d)
        # kwargs of nn.ConvNd
        self.conv = self.ConvType(
            in_channels=c1, out_channels=c2, kernel_size=k,
            stride=s, padding=padding, groups=g, dilation=d, bias=False,
        )
        self.bn = self.BnType(c2)
        self.act = act() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    @classmethod
    def create_mlp(cls, c1, c2s, k=1, linear_output=False):
        """ Create MLP. """
        layers = nn.Sequential()
        for c2 in (c2s[:-1] if linear_output else c2s):
            layers.append(cls(c1, c2, k=k, s=1))
            c1 = c2
        if linear_output:
            layers.append(cls.ConvType(c1, c2s[-1], kernel_size=1))
        layers.c2 = c2s[-1]
        return layers


class ConvBnAct1d(_ConvBnActNd):
    ConvType = nn.Conv1d
    BnType = nn.BatchNorm1d


class ConvBnAct2d(_ConvBnActNd):
    ConvType = nn.Conv2d
    BnType = nn.BatchNorm2d


class ConvBnAct3d(_ConvBnActNd):
    ConvType = nn.Conv3d
    BnType = nn.BatchNorm3d


class ELA(nn.Module):

    def __init__(self, c1, c2, e=0.5, n=3):
        super().__init__()
        self.c2 = c2
        c_ = make_divisible(c2 * e, divisor=4)
        self.ela1 = ConvBnAct2d(c1, c_, 1)
        self.ela2 = ConvBnAct2d(c1, c_, 1)
        self.elan = nn.ModuleList(
            nn.Sequential(ConvBnAct2d(c_, c_, 3),
                          ConvBnAct2d(c_, c_, 3)) for _ in range(n)
        )
        self.elap = ConvBnAct2d(c_ * (n + 2), c2, 1)

    def forward(self, x):
        y = [self.ela1(x), self.ela2(x)]
        for m in self.elan: y.append(m(y[-1]))
        return self.elap(torch.cat(y, 1))


class CspOSA(nn.Module):

    def __init__(self, c1, c2, e=0.5, n=4):
        super().__init__()
        self.c2 = c2
        c_ = make_divisible(c2 * e, divisor=4)
        n = max(2, n)
        self.osa1 = ConvBnAct2d(c1, c_ * 2, 1)
        self.osa2 = ConvBnAct2d(c1, c_ * 2, 1)
        self.osa3 = ConvBnAct2d(c_ * 2, c_, 3)
        self.osan = nn.ModuleList(
            ConvBnAct2d(c_, c_, 3) for _ in range(n - 1)
        )
        self.osap = ConvBnAct2d(c_ * (n + 4), c2, 1)

    def forward(self, x):
        y = [self.osa1(x), self.osa2(x)]
        y.append(self.osa3(y[-1]))
        for m in self.osan: y.append(m(y[-1]))
        return self.osap(torch.cat(y, 1))


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, s=1, g=1, d=1, e=0.25):
        super().__init__()
        self.c2 = c2
        c_ = make_divisible(c2 * e, divisor=4)
        self.btn1 = ConvBnAct2d(c1, c_, 1)
        self.btn2 = ConvBnAct2d(c_, c_, 3, s, g, d)
        self.btn3 = ConvBnAct2d(c_, c2, 1, act=None)
        self.downs = nn.Identity() if c1 == c2 and s == 1 else ConvBnAct2d(c1, c2, 1, s, act=None)
        self.act = self.btn1.act

    def forward(self, x):
        return self.act(self.downs(x) + self.btn3(self.btn2(self.btn1(x))))


class SEReLU(nn.Module):
    """ Squeeze-and-Excitation Block"""

    def __init__(self, c1, r=16):
        super().__init__()
        self.c2 = c1
        c_ = max(4, c1 // r)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c_, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(c_, c1, 1, bias=True),
            nn.Sigmoid()
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x * self.se(x))


class Upsample(nn.Upsample):

    def __init__(self, s=2, mode="nearest"):
        super().__init__(scale_factor=s, mode=mode)


class GeM(nn.Module):
    """ Generalized-Mean Pooling"""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp_min(1e-6).pow(self.p), 1).pow(1. / self.p)


class CossimBce(nn.Module):

    def __init__(self,
                 s: float = 10.):
        super().__init__()
        s = torch.tensor(s, dtype=torch.float32)
        self.t = nn.Parameter(s)
        self.b = nn.Parameter(-s)

    def example_label(self, B, G) -> torch.Tensor:
        z = torch.full([G] * 2, -1, dtype=torch.int64)
        z.fill_diagonal_(1)
        return z[None].repeat(B, 1, 1).to(self.b.device)

    def extra_repr(self) -> str:
        return "t=%.2f, b=%.2f" % (self.t.item(), self.b.item())

    def forward(self, z, x1, x2=None):
        """
        :param z: [B, N, M] labels, {1, 0, -1}
        :param x1: [B, N, C]
        :param x2: [B, M, C]
        """
        if x2 is None: x2 = x1
        cos_sim = torch.cosine_similarity(x1.unsqueeze(-2), x2.unsqueeze(-3), dim=-1)
        y = z.to(torch.float32) * (self.t * cos_sim - self.b)  # [B, N, M]
        mask = z != 0
        loss = - F.logsigmoid(y)[mask].mean()
        return dict(loss=loss, cos_sim=cos_sim, mask=mask)


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)

    loss_fn = CossimBce().cuda()
    print(loss_fn)
    x1 = torch.randn(2, 3, 8).cuda()
    x2 = torch.randn(2, 3, 8).cuda()
    y = torch.eye(3).cuda()[None].repeat(2, 1, 1)
    y[y == 0] = -1

    loss = loss_fn(y, x1, x2)
    print(loss)
