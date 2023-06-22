import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

red = 'orangered'
orange = 'orange'
yellow = 'yellow'
green = 'greenyellow'
cyan = 'aqua'
blue = 'deepskyblue'
purple = 'mediumpurple'
pink = 'violet'


def standard_coord(*args, zero_p=True):
    ''' 显示标准轴
        args: subplot 参数'''
    fig = plt.subplot(*args)
    for key in 'right', 'top':
        fig.spines[key].set_color('None')
    if zero_p:
        for key in 'left', 'bottom':
            fig.spines[key].set_position(('data', 0))
    return fig


def boxplot(dataset, labels, colors):
    ''' 绘制箱线图'''
    bp = plt.boxplot(dataset, labels=labels)
    for i, color in enumerate(colors):
        bp['boxes'][i].set(color=color, linewidth=1.5)
        bp['medians'][i].set(color=color, linewidth=2.1)


def bar2d(dataset, xticks=None, labels=None, colors=None, alpha=1):
    x = np.arange(dataset.shape[1])
    bias = np.linspace(-.5, .5, dataset.shape[0] + 2)[1:-1]
    w = .7 * (.5 - bias[-1])
    # 处理缺失值信息
    labels = [None] * len(bias) if labels is None else labels
    colors = [None] * len(bias) if colors is None else colors
    for i, y in enumerate(dataset):
        plt.bar(x + bias[i], y, width=w, label=labels[i], color=colors[i], alpha=alpha)
    # 绘制标签信息
    if any(labels): plt.legend()
    if xticks: plt.xticks(x, xticks)


def bar3d(figure, center, a, b, h, color=None):
    ''' 绘制 3D 柱状图
        center: 圆柱中心点
        a, b, h: 柱体边长'''
    x = np.array([-a, a]) / 2 + center[0]
    y = np.array([-b, b]) / 2 + center[1]
    z = np.array([-h, h]) / 2 + center[2]
    x, y, z = np.meshgrid(x, y, z)
    filled = np.ones([1] * 3)
    figure.voxels(x, y, z, filled,
                  facecolors=color, edgecolors=color)
    return figure


def hotmap(array, cmap='Blues', annotate=10, colorbar=False,
           xticks=None, yticks=None, xlabel=None, ylabel=None):
    # 去除坐标轴
    fig = plt.subplot()
    for key in 'right', 'top', 'left', 'bottom':
        fig.spines[key].set_color('None')
    fig.xaxis.set_ticks_position('top')
    fig.xaxis.set_label_position('top')
    # 显示热力图
    plt.imshow(array, cmap=cmap)
    if colorbar: plt.colorbar()
    # 标注数据信息
    for i, row in enumerate(array):
        for j, item in enumerate(row):
            plt.annotate(f'{item:.2f}', (j - .1, i + .05), size=annotate)
    # 坐标轴标签
    plt.xticks(range(len(array)), xticks)
    plt.yticks(range(len(array[0])), yticks, rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def heat_img(img, heat, cmap=cv2.COLORMAP_JET):
    if heat.dtype != np.uint8: heat = np.uint8(np.round(heat * 255))[..., None].repeat(3, -1)
    heat = cv2.applyColorMap(heat, colormap=cmap)
    return cv2.addWeighted(img, .5, heat, .5, 0)


def torch_show(img, delay=0):
    ''' img: [B, C, H, W] or [C, H, W]'''
    assert img.dtype == torch.uint8
    img = img.data.numpy()
    img = img[None] if img.ndim == 3 else img
    img = img.transpose(0, 2, 3, 1)[..., ::-1]
    for i in img:
        cv2.imshow('debug', i)
        cv2.waitKey(delay)


class ParamUtilization:

    def __new__(cls, model: nn.Module, path='model', sample=4,
                norm_kernel=False, decimals=3):
        self = object.__new__(cls)
        self.sample = sample
        self.norm_kernel = norm_kernel
        self.round = lambda x: list(map(lambda i: round(i, decimals), x.tolist()))
        self.result = {}
        self(model, path)
        return pd.DataFrame(self.result).T

    def __call__(self, model, path):
        # 如果有属性 weight 则计算参数利用率
        if hasattr(model, 'weight'):
            weight = model.weight.data
            c2, *not_1d = weight.shape
            info = {'c2': c2}
            if not_1d:
                # 如果是卷积核
                if self.norm_kernel and len(not_1d) == 3 and not_1d[-1] != 1:
                    wc = weight.clone()
                    k_size, norm_k = not_1d[-1], []
                    for i in range((k_size - 1) // 2, -1, -1):
                        k1, k2 = k_size - i * 2, max(0, k_size - (i + 1) * 2)
                        # 计算归一化权值
                        norm_k.append(((wc[..., i:i + k1, i:i + k1]).sum() / (k1 ** 2 - k2 ** 2)).abs().item())
                        wc[..., i:-i, i:-i] *= 0
                    norm_k = np.array(norm_k, dtype=np.float32)
                    info['norm-kernel'] = self.round(norm_k / (norm_k.mean() + 1e-6))
                # 计算相对稀疏度
                weight = weight.view(c2, -1)
                norm = torch.norm(weight, dim=-1)
                info['norm-mean'] = norm.mean().item()
                norm /= info['norm-mean']
                # 计算输出通道的余弦相似度
                cosine = torch.cosine_similarity(
                    weight[None], weight[:, None], dim=-1, eps=1e-3
                ).abs() - torch.eye(c2).half().to(weight.device)
                cosine = 1 - cosine.max(dim=0)[0]
                # 定义主导度为: sqrt(cossim) × norm
                info['domiance'] = self.round(torch.sort(norm * cosine.sqrt()
                                                         )[0][::self.sample].cpu().data.numpy())
                self.result[path] = info
        # 递归搜索
        else:
            for k, m in model._modules.items(): self(m, f'{path}[{k}:{type(m).__name__}]')
