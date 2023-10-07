import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn


def heat_img(img, heat, cmap=cv2.COLORMAP_JET):
    if heat.dtype != np.uint8: heat = np.uint8(np.round(heat * 255))[..., None].repeat(3, -1)
    heat = cv2.applyColorMap(heat, colormap=cmap)
    return cv2.addWeighted(img, .5, heat, .5, 0)


def torch_show(img, delay=0):
    ''' :param img: [B, C, H, W] or [C, H, W]'''
    assert img.dtype == torch.uint8
    img = img.data.numpy()
    img = img[None] if img.ndim == 3 else img
    img = img.transpose(0, 2, 3, 1)[..., ::-1]
    for i in img:
        cv2.imshow('debug', i)
        cv2.waitKey(delay)


class LossLandScape:

    def __init__(self, w=1., dpi=20):
        self.dpi = dpi + ((dpi + 1) & 1)
        w = np.linspace(-w, w, self.dpi)
        self.coord = np.stack(np.meshgrid(w, w), axis=-1)

    def process(self, m0, m1, m2, m3, m4):
        ms = [m - m0 for m in (m1, m2, m3, m4)]
        for x, y in self.coord.reshape(-1, 2):
            m = m0
            if x: m = m + abs(x) * ms[x > 0]
            if y: m = m + abs(y) * ms[2 + (y > 0)]
            yield m

    def plot(self, losses, cmap='Blues'):
        losses = np.array(losses).reshape(self.dpi, self.dpi)
        fig = plt.subplot(projection='3d')
        fig.plot_surface(self.coord[..., 0], self.coord[..., 1], losses, cmap=cmap)
        plt.show()


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
