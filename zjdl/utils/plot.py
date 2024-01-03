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
    def _parse_kernel(cls, weight, info):
        c2, *not_1d = weight.shape
        # 如果是 n×n 卷积核, 由里到外计算 卷积核核环 的平均范数
        if len(not_1d) == 3 and not_1d[-1] != 1 and not_1d[-1] == not_1d[-2]:
            wc = np.linalg.norm(weight, 2, axis=(0, 1))
            k_size, norm_k = not_1d[-1], []
            # 外核左上角元素的 横纵坐标
            for i in range((k_size - 1) // 2, -1, -1):
                # 外核、内核 的核尺寸
                k1 = k_size - i * 2
                k2 = max(0, k1 - 2)
                # 计算平均范数
                norm_k.append(np.abs((wc[i:i + k1, i:i + k1]).sum() / (k1 ** 2 - k2 ** 2)).item())
                wc[i:-i, i:-i] *= 0
            norm_k = np.array(norm_k, dtype=np.float32)
            info['norm-kernel'] = cls._round(norm_k / (norm_k.mean() + 1e-6))

    @classmethod
    def _parse_weight(cls, weight) -> dict:
        weight = weight.float().cpu().numpy()
        c2, *not_1d = weight.shape
        if not_1d:
            info = {'c2': c2}
            cls._parse_kernel(weight, info)
            weight = weight.reshape(c2, -1)
            # 计算权重向量二范数 norm
            norm = np.linalg.norm(weight, 2, axis=-1)
            # 根据 norm, 对 weight, norm 进行排序
            i = np.argsort(norm)
            norm = norm[i]
            weight = weight[i]
            # 对权重向量单位化, 以内积作为余弦相似度
            vec = weight / norm[:, None]
            cos = (np.abs(vec @ vec.T) - np.eye(c2)).clip(max=1)
            sin = np.sqrt(1 - cos ** 2)
            # 差异程度 (取最小): 正弦值 * norm 相对大小
            diff = sin * norm[:, None] / norm
            score = np.array([diff[i, i:].min() for i in range(c2 - 1)])
            # y = np.linalg.svd(weight)[1]
            # y /= y[0]
            info['score'] = cls._round(np.sort(score))
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
    def export(cls, result, project, filt=None, show=False, group_lv=1, sep='.', limit=25, **vplot_kwd):
        ''' :param result: parse 方法输出的结果 / 文件路径
            :param project: 项目目录
            :param filt: 过滤器, 筛选输出的 module
            :param show: 是否显示图像
            :param group_lv: module 进行分组的层级
            :param vplot_kwd: violinplot 的参数'''
        from mod.zjplot import violinplot, rand_colors
        # 创建项目目录, 输出 csv
        project.mkdir(parents=True, exist_ok=True)
        name = project.name
        csv = project / f'{name}.csv'
        if isinstance(result, pd.DataFrame):
            result.to_csv(csv)
        else:
            result = pd.read_csv(result, index_col=0)
            trans = lambda x: x if pd.isna(x) else eval(str(x))
            for k in result.columns:
                if str(result[k].dtype) == 'object':
                    result[k] = result[k].apply(trans)
        # 对神经网络中的层进行分组
        k2i = lambda k: sep.join(k.split(sep)[:group_lv + 1])
        groups = list(map(k2i, result.index))
        for i in range(len(groups) - 1, 0, -1):
            if groups[i] == groups[i - 1]: groups.pop(i)
        colors = rand_colors(len(groups))
        # 绘图相关参数设定
        if filt: result = result.loc[filter(filt, result.index)]
        limit = limit if limit else len(result)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['figure.figsize'] = [.8 + 0.46 * (limit + 1), 6.4]
        ymin = result['score'].apply(min).min()
        ymax = result['score'].apply(max).max()
        # 分页读取 result
        for i in tqdm(range(int(np.ceil(len(result) / limit))), desc='exporting plots'):
            tmp = result.iloc[i * limit: (i + 1) * limit]
            plt.clf()
            plt.ylabel('score')
            # 根据分组分配颜色
            violinplot(tmp['score'], labels=list(tmp.index),
                       colors=[colors[groups.index(k2i(k))] for k in tmp.index], xrotate=90, **vplot_kwd)
            # 设置上下限, 布局优化
            plt.xlim([0, limit + 1]), plt.ylim(ymin, ymax)
            plt.grid(), plt.tight_layout()
            plt.show() if show else (plt.savefig(project / f'{name}{i}.png'), plt.close())
