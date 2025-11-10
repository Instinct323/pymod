import logging
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOGGER = logging.getLogger("utils")

sum_ = lambda x: sum(x[1:], x[0])
BilinearResize = partial(F.interpolate, mode="bilinear", align_corners=False)
make_divisible = lambda x, divisor=4: np.maximum(np.round(x / divisor).astype(np.int64), 1) * divisor

# Registry
module_required = {k: [] for k in ("c1,c2", "c1", "n")}


def register_module(*args):
    def solver(cls):
        for k in args:
            module_required[k].append(cls)
        return cls

    return solver


def auto_pad(k, s=1, d=1):
    # (k - 1) // 2 * d: 使卷积中心位移到图像的 [0, 0]
    # (s - 1) // 2: 使卷积中心位移到 [s/2, s/2]
    return max(0, (k - 1) // 2 * d - (s - 1) // 2)


tuple(map(register_module("c1,c2"), (nn.Linear, nn.Conv1d, nn.Conv2d)))
tuple(map(register_module("c1"), (nn.BatchNorm2d, nn.LayerNorm)))


@register_module("c1")
class BatchNorm(nn.BatchNorm2d):

    def __init__(self, c1, s=1):
        super().__init__(c1)
        self.c2 = c1
        self.s = s

    def forward(self, x):
        return super().forward(x if self.s == 1 else x[..., ::self.s, ::self.s])

    def unpack(self, detach=False):
        mean, bias = self.running_mean, self.bias
        std = (self.running_var + self.eps).float().sqrt().to(mean)
        weight = self.weight / std
        eq_param = weight, bias - weight * mean
        return tuple(map(lambda x: x.data, eq_param)) if detach else eq_param


@register_module("c1,c2")
class Conv2d(nn.Conv2d):
    """ Conv - BN - Act"""
    deploy = property(lambda self: isinstance(self.bn, nn.Identity))

    def __init__(self, c1, c2, k=3, s=1, g=1, d=1,
                 act: Optional[nn.Module] = nn.ReLU, ctrpad=True):
        self.c2 = c2
        assert k & 1, "The convolution kernel size must be odd"
        # depthwise separable convolution
        if g == "dw":
            g = c1
            assert c1 == c2, "Failed to create DWConv"
        # kwargs of nn.Conv2d
        self.cfg_ = dict(
            in_channels=c1, out_channels=c2, kernel_size=k,
            stride=s, padding=auto_pad(k, s if ctrpad else 1, d), groups=g, dilation=d
        )
        super().__init__(**self.cfg_, bias=False)
        self.bn = BatchNorm(c2)
        self.act = act() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(super().forward(x)))

    @classmethod
    def reparam(cls, model: nn.Module):
        for m in filter(lambda m: isinstance(m, cls) and not m.deploy, model.modules()):
            kernel = m.weight.data.clone()
            bn_w, bn_b = m.bn.unpack(detach=True)
            # Merging nn.Conv2d with BatchNorm
            m.weight.data, m.bias = kernel * bn_w.view(-1, 1, 1, 1), nn.Parameter(bn_b)
            m.bn = nn.Identity()


@register_module("c1,c2")
class RepConv(nn.Module):
    deploy = property(lambda self: isinstance(self.rep, nn.Conv2d))

    def __init__(self, c1, c2, k=(0, 1, 3), s=1, g=1, d=1,
                 act: Optional[nn.Module] = nn.ReLU):
        """ :param k: 卷积核尺寸, 0 表示恒等映射 """
        super().__init__()
        self.c2 = c2
        self.rep = nn.ModuleList()
        assert len(k) > 1, "RepConv with a single branch is illegal"
        for k in sorted(k):
            # Identity
            if k == 0:
                assert c1 == c2, "Failed to add the identity mapping branch"
                self.rep.append(BatchNorm(c2, s=s))
            # nn.Conv2d + BatchNorm
            elif k > 0:
                assert k & 1, f"The convolution kernel size {k} must be odd"
                self.rep.append(Conv2d(c1, c2, k=k, s=s, g=g, d=d, act=None, ctrpad=False))
            else:
                raise AssertionError(f"Wrong kernel size {k}")
        # Activation
        self.act = act() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.rep(x) if self.deploy else sum_(tuple(m(x) for m in self.rep)))

    @classmethod
    def reparam(cls, model: nn.Module):
        Conv2d.reparam(model)
        # 查询模型的所有子模型, 对 RepConv 进行合并
        for m in filter(lambda m: isinstance(m, cls) and not m.deploy, model.modules()):
            src, cfg = m.rep[-1].weight, m.rep[-1].cfg_
            conv = nn.Conv2d(**cfg, bias=True).to(src)
            mlist, m.rep = m.rep, conv
            (c2, c1g, k, _), g = conv.weight.shape, conv.groups
            # nn.Conv2d 参数置零
            nn.init.constant_(conv.weight, 0)
            nn.init.constant_(conv.bias, 0)
            for branch in mlist:
                # BatchNorm
                if isinstance(branch, BatchNorm):
                    w, b = branch.unpack(detach=True)
                    conv.weight.data[..., k // 2, k // 2] += torch.eye(c1g).repeat(g, 1).to(src) * w[:, None]
                # Conv
                else:
                    p = (k - branch.kernel_size[0]) // 2
                    w, b = branch.weight.data, branch.bias.data
                    conv.weight.data += F.pad(w, (p,) * 4)
                conv.bias.data += b


@register_module("c1,c2", "n")
class ELA(nn.Module):

    def __init__(self, c1, c2, e=0.5, n=3):
        super().__init__()
        self.c2 = c2
        c_ = make_divisible(c2 * e, divisor=4)
        self.ela1 = Conv2d(c1, c_, 1)
        self.ela2 = Conv2d(c1, c_, 1)
        self.elan = nn.ModuleList(
            nn.Sequential(Conv2d(c_, c_, 3),
                          Conv2d(c_, c_, 3)) for _ in range(n)
        )
        self.elap = Conv2d(c_ * (n + 2), c2, 1)

    def forward(self, x):
        y = [self.ela1(x), self.ela2(x)]
        for m in self.elan: y.append(m(y[-1]))
        return self.elap(torch.cat(y, 1))


@register_module("c1,c2", "n")
class CspOSA(nn.Module):

    def __init__(self, c1, c2, e=0.5, n=4):
        super().__init__()
        self.c2 = c2
        c_ = make_divisible(c2 * e, divisor=4)
        n = max(2, n)
        self.osa1 = Conv2d(c1, c_ * 2, 1)
        self.osa2 = Conv2d(c1, c_ * 2, 1)
        self.osa3 = Conv2d(c_ * 2, c_, 3)
        self.osan = nn.ModuleList(
            Conv2d(c_, c_, 3) for _ in range(n - 1)
        )
        self.osap = Conv2d(c_ * (n + 4), c2, 1)

    def forward(self, x):
        y = [self.osa1(x), self.osa2(x)]
        y.append(self.osa3(y[-1]))
        for m in self.osan: y.append(m(y[-1]))
        return self.osap(torch.cat(y, 1))


@register_module("c1,c2")
class Bottleneck(nn.Module):

    def __init__(self, c1, c2, s=1, g=1, d=1, e=0.25):
        super().__init__()
        self.c2 = c2
        c_ = make_divisible(c2 * e, divisor=4)
        self.btn1 = Conv2d(c1, c_, 1)
        self.btn2 = Conv2d(c_, c_, 3, s, g, d)
        self.btn3 = Conv2d(c_, c2, 1, act=None)
        self.downs = nn.Identity() if c1 == c2 and s == 1 else Conv2d(c1, c2, 1, s, act=None)
        self.act = self.btn1.act

    def forward(self, x):
        return self.act(self.downs(x) + self.btn3(self.btn2(self.btn1(x))))


class Shortcut(nn.Module):

    def forward(self, x):
        return sum_(x)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


@register_module("c1")
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


class AvgPool(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.mean(dim=self.dims)
