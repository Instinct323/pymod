from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


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
    decimals = 3

    @classmethod
    def _round(cls, x):
        return list(map(lambda i: round(i, cls.decimals), x.tolist()))

    @classmethod
    def _parse_weight(cls, weight) -> dict:
        weight = weight.float().cpu()
        c2, *not_1d = weight.shape
        if not_1d:
            info = {'c2': c2}
            # 如果是 n×n 卷积核, 由里到外计算 卷积核核环 的平均范数
            if len(not_1d) == 3 and not_1d[-1] != 1 and not_1d[-1] == not_1d[-2]:
                wc = torch.norm(weight, dim=(0, 1))
                k_size, norm_k = not_1d[-1], []
                # 外核左上角元素的 横纵坐标
                for i in range((k_size - 1) // 2, -1, -1):
                    # 外核、内核 的核尺寸
                    k1 = k_size - i * 2
                    k2 = max(0, k1 - 2)
                    # 计算平均范数
                    norm_k.append(((wc[i:i + k1, i:i + k1]).sum() / (k1 ** 2 - k2 ** 2)).abs().item())
                    wc[i:-i, i:-i] *= 0
                norm_k = torch.tensor(norm_k, dtype=torch.float32)
                info['norm-kernel'] = cls._round(norm_k / (norm_k.mean() + 1e-6))
            weight = weight.view(c2, -1)
            # 计算权重向量二范数 norm
            norm = torch.norm(weight, dim=-1)
            info['norm-mean'] = norm.mean().item()
            # 根据 norm, 对 weight, norm 进行排序
            i = torch.argsort(norm)
            norm = norm[i]
            weight = weight[i]
            # 计算余弦相似度
            cos = torch.cosine_similarity(weight, weight[:, None], dim=-1, eps=1e-6).abs() - torch.eye(c2)
            # 信息损失 (取最小): (1 - 余弦值) * norm 相对大小
            norm_loss = (1 - cos) * norm[:, None] / norm
            score = torch.tensor([norm_loss[i, i:].min() for i in range(c2 - 1)])
            info['score'] = cls._round(torch.sort(score)[0])
            return info

    @classmethod
    def parse(cls, model_or_sdit, **export_kwd):
        result = {}

        if isinstance(model_or_sdit, nn.Module):
            def solve(model, path):
                # 如果有属性 weight 则计算参数利用率
                if hasattr(model, 'weight'):
                    info = cls._parse_weight(model.weight.data)
                    if info: result[path[1:]] = info
                # 递归搜索
                else:
                    for k, m in model._modules.items(): solve(m, f'{path}.{k}[{type(m).__name__}]')

            solve(model_or_sdit, '')

        elif isinstance(model_or_sdit, OrderedDict):
            suffix = '.weight'
            for k, v in model_or_sdit.items():
                if k.endswith(suffix):
                    info = cls._parse_weight(v)
                    if info: result[k.rstrip(suffix)] = info

        else:
            raise TypeError(f'Incorrect argument type {type(model_or_sdit).__name__}')

        result = pd.DataFrame(result).T
        cls.export(result, **export_kwd)
        return result

    @classmethod
    def export(cls, result, project, show=False, group_lv=1, sep='.', limit=25, **vplot_kwd):
        from mod.zjplot import rand_colors, violinplot
        # 创建项目目录, 输出 csv
        project.mkdir(parents=True, exist_ok=True)
        (project / 'pu.csv').csv(result)
        # 绘图相关参数设定
        limit = limit if limit else len(result)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['figure.figsize'] = [.8 + 0.46 * (limit + 1), 6.4]
        ymin = min(map(min, result['score']))
        ymax = max(map(max, result['score']))
        # 对神经网络中的层进行分组
        k2i = lambda k: sep.join(k.split(sep)[:group_lv + 1])
        groups = tuple({k2i(k) for k in result.index})
        colors = rand_colors(len(groups))
        # 分页读取 result
        for i in tqdm(range(int(np.ceil(len(result) / limit))), desc='exporting plots'):
            tmp = result.iloc[i * limit: (i + 1) * limit]
            plt.ylabel('score')
            # 根据分组分配颜色
            violinplot(tmp['score'], labels=list(tmp.index),
                       colors=[colors[groups.index(k2i(k))] for k in tmp.index], xrotate=90, **vplot_kwd)
            # 设置上下限, 布局优化
            plt.xlim([0, limit + 1]), plt.ylim(ymin, ymax)
            plt.grid(), plt.tight_layout()
            plt.show() if show else (plt.savefig(project / f'pu{i}.jpg'), plt.close())
