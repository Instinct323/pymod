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
        apply_width_multiplier(c, w, divisor) if isinstance(c, list) else make_divisible(c * w, divisor=divisor)
        for c in channels
    ]


def fuse_modules(model: nn.Module):
    for m in filter(lambda m: isinstance(m, LinearBnAct) and not m.deploy, model.modules()):
        torch.ao.quantization.fuse_modules(m, ["linear", "bn"], inplace=True)


class LinearBnAct(nn.Module):
    """ Linear - BN - Act """
    deploy = property(lambda self: isinstance(self.bn, nn.Identity))
    LinearType = nn.Linear
    BnType = nn.BatchNorm1d

    def __init__(self, c1, c2, act: Optional[nn.Module] = nn.ReLU, **linear_kwargs):
        super().__init__()
        self.c2 = c2
        self.linear_kwargs = linear_kwargs
        self.linear = self.LinearType(c1, c2, bias=False, **linear_kwargs)
        self.bn = self.BnType(c2)
        self.act = act() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.linear(x)))

    @classmethod
    def create_mlp(cls, c1, c2s, linear_output=False, **kwargs):
        """ Create MLP. """
        layers = nn.Sequential()
        for c2 in (c2s[:-1] if linear_output else c2s):
            layers.append(cls(c1, c2, **kwargs))
            c1 = c2
        if linear_output:
            layers.append(cls.LinearType(c1, c2s[-1], bias=True, **layers[-1].linear_kwargs))
        layers.c2 = c2s[-1]
        return layers


class _ConvBnActNd(LinearBnAct):
    """ Conv - BN - Act """

    @staticmethod
    def auto_pad(k, s=1, d=1):
        # (k - 1) // 2 * d: 1st-center -> [0, 0]
        # (s - 1) // 2: 1st-center -> [s/2, s/2]
        return max(0, (k - 1) // 2 * d - (s - 1) // 2)

    def __init__(self, c1, c2, k=3, s=1, g=1, d=1,
                 act: Optional[nn.Module] = nn.ReLU, ctrpad=True):
        assert k & 1, "The convolution kernel size must be odd"
        padding = self.auto_pad(k, s if ctrpad else 1, d)
        super().__init__(c1, c2, act=act,
                         kernel_size=k, stride=s, padding=padding, groups=g, dilation=d)


class ConvBnAct1d(_ConvBnActNd):
    LinearType = nn.Conv1d
    BnType = nn.BatchNorm1d


class ConvBnAct2d(_ConvBnActNd):
    LinearType = nn.Conv2d
    BnType = nn.BatchNorm2d


class ConvBnAct3d(_ConvBnActNd):
    LinearType = nn.Conv3d
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


class CossimBCE(nn.Module):

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

    def forward(self, z, x1, x2=None) -> dict[str, torch.Tensor]:
        """
        :param z: [B, N, M] labels, {1, 0, -1}
        :param x1: [B, N, C]
        :param x2: [B, M, C]
        """
        if x2 is None: x2 = x1
        # require at least one positive and one negative sample
        B_mask = torch.any(z == -1, dim=(1, 2)) & torch.any(z == 1, dim=(1, 2))
        z, x1, x2 = z[B_mask], x1[B_mask], x2[B_mask]
        B = z.shape[0]

        cos_sim = torch.cosine_similarity(x1.unsqueeze(-2), x2.unsqueeze(-3), dim=-1)
        y = z.to(torch.float32) * (self.t * cos_sim - self.b)  # [B, N, M]
        loss = - F.logsigmoid(y)[z != 0].mean()

        ret = {"loss": loss, "batch_size": B}
        if not self.training:
            mask_pos, mask_neg = z == 1, z == -1
            ret["cos_diff"] = sum(cos_sim[b][mask_pos[b]].mean() - cos_sim[b][mask_neg[b]].mean() for b in range(B)) / B
        return ret


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)

    x = torch.randn(2, 3, 2)
    model = ConvBnAct1d.create_mlp(3, [16, 32, 64], linear_output=True).eval()
    print(model)
    fuse_modules(model)
    print(model)

    y = model(x)
    print(y.shape)
