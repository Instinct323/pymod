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
    def create_mlp(cls, c1, c2s, k=1, linear_output=False, **kwargs):
        """ Create MLP. """
        layers = nn.Sequential()
        for c2 in (c2s[:-1] if linear_output else c2s):
            layers.append(cls(c1, c2, k=k, s=1, **kwargs))
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


class GroupAvgPool1d(nn.Module):

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, x, y) -> dict[str, torch.Tensor]:
        """
        :param x: [B, N, C]
        :param y: [B, N]
        :returns: feature[B, G, C], mask[B, G]
        """
        y = y.clone()
        y[y < 0] = self.g
        x = x.unsqueeze(-2)     # [B, N, 1, C]
        one_hot = F.one_hot(y)[..., :-1].to(x.device).unsqueeze(-1)  # [B, N, G, 1]
        return dict(feature=(x * one_hot).mean(dim=1), mask=torch.any(one_hot, dim=(1, 3)))


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

    def forward(self, z, x1, x2=None) -> dict[str, torch.Tensor]:
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
    B, N, g = 4, 10, 8

    x = torch.rand([B, 16, N])
    y = torch.randint(-1, g, [B, N])

    pool = GroupAvgPool1d(g)
    ret = pool(x, y)
    print(ret)
