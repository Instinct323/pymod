import warnings
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# 重新封装 sum 函数, 以减少加 0 时产生的计算
sum_ = lambda x: sum(x[1:], x[0])
# 重新封装插值函数
BilinearResize = partial(F.interpolate, mode='bilinear', align_corners=False)
# 配合 YamlModel 类使用, 使其可以自动填充参数
module_required = {k: [] for k in ('c1,c2', 'c1', 'n')}
# 等比数列
logspace = lambda start, stop, n: np.logspace(np.log10(start), np.log10(stop), 4)
make_divisible = lambda x, divisor=4: np.maximum(np.round_(x / divisor).astype(np.int64), 1) * divisor


def register_module(*args):
    def solver(cls):
        for k in args:
            module_required[k].append(cls)
        return cls

    return solver


class LOCAL:
    hw = None  # Size of the feature map before patch embedding
    pmask = None  # Mask for patches


def auto_pad(k, s=1, d=1):
    # (k - 1) // 2 * d: 使卷积中心位移到图像的 [0, 0]
    # (s - 1) // 2: 使卷积中心位移到 [s/2, s/2]
    return (k - 1) // 2 * d - (s - 1) // 2


def vec2img(x):
    B, L, C = map(int, x.shape)
    # x[B, L, C] -> x[B, C, H, W]
    return x.transpose(1, 2).view(-1, C, *LOCAL.hw)


def img2vec(x, in_backbone=True):
    if in_backbone:
        B, C, *LOCAL.hw = map(int, x.shape)
    # x[B, C, H, W] -> x[B, L, C]
    return x.flatten(start_dim=2).transpose(1, 2)


def hard_softmax(x, dim=-1):
    y_soft = x.softmax(dim)
    y_hard = torch.scatter(torch.zeros_like(x), dim=dim,
                           index=y_soft.argmax(dim, keepdim=True), value=1.0)
    return y_hard - y_soft.detach() + y_soft


def make_grid(w, h, percent=True, center=False, axis=-1):
    coord = np.stack(np.meshgrid(*map(np.arange, (w, h))), axis=axis).astype(np.float32)
    if center: coord += .5
    return coord / np.array([w, h]) if percent else coord


def set_groups(self: nn.Conv2d, g: int, strict: bool = True):
    if self.groups != g:
        # 检查新的分组是否会破坏权重
        if strict and self.groups % g:
            warnings.warn(f'Because groups({self.groups}) % g({g}) != 0, some of the weight will be lost')
        # 转换为标准卷积
        c2, c1 = self.out_channels // self.groups, self.in_channels // self.groups
        w = torch.zeros_like(self.weight).repeat(1, self.groups, 1, 1)
        for i in range(self.groups):
            c2s, c1s = i * c2, i * c1
            w[c2s: c2s + c2, c1s: c1s + c1] = self.weight.data[c2s: c2s + c2]
        if g != 1:
            w_full = w
            # 检查新的分组是否合法
            c2, c1 = self.out_channels // g, self.in_channels // g
            assert c1 * g == self.in_channels, f'in_channels({self.in_channels}) must be divisible by groups({g})'
            assert c2 * g == self.out_channels, f'out_channels({self.out_channels}) must be divisible by groups({g})'
            # 转换为新的分组卷积
            w = torch.zeros_like(w_full[:, :1]).repeat(1, c1, 1, 1)
            for i in range(g):
                c2s, c1s = i * c2, i * c1
                w[c2s: c2s + c2] = w_full[c2s: c2s + c2, c1s: c1s + c1]
        # 原地修改权重
        self.groups = g
        self.weight = nn.Parameter(w)
