import collections
import copy
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .utils import LOGGER

to_2tuple = lambda x: x if isinstance(x, collections.abc.Iterable) and not isinstance(x, str) else (x,) * 2


class Activation(nn.Module):
    identity = False

    def forward(self, x):
        return x if self.identity else self.Func.apply(x)

    class Func(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            return (x / x.max()).sqrt()

        @staticmethod
        def backward(ctx, gy):
            return gy / gy.mean()


class ReceptiveField:
    n_sample = 8

    def __init__(self,
                 model: nn.Module,
                 tar_layer: Union[int, nn.Module],
                 img_size: Union[int, Tuple[int, int]],
                 in_channels: int = 3,
                 use_cuda: bool = False):
        # 注册前向传播的挂钩
        tar_layer = model[tar_layer] if isinstance(tar_layer, int) else tar_layer
        tar_layer.register_forward_hook(
            lambda module, x, y: setattr(self, '_fmap', y)
        )
        # 获得该模型的 stride
        img_size = to_2tuple(img_size)
        model(torch.zeros([1, in_channels, *img_size]))
        assert self._fmap.dim() == 4, f'Invalid selection of tar_layer {type(tar_layer)}'
        fmap_size, self._fmap = self._fmap.shape[2:], None
        stride = tuple(round(i / f) for i, f in zip(img_size, fmap_size))
        # 将特征图尺寸更正为奇数
        self.img_size = [in_channels] + [(f - ((f + 1) & 1)) * s for f, s in zip(fmap_size, stride)]
        try:
            self.model = copy.deepcopy(model).eval()
        except:
            self.model = model.eval()
            LOGGER.warning('Fail to deep copy the model, the model will be modified in place')
        # 原地替换激活函数
        self.replace(self.model)
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def compare(self, theoretical=True, original=True, state_dict=None, **imshow_kw):
        ''' :param state_dict: 完成训练的模型权值'''
        task = []
        if original: task.append(('Before Training', self.effective()))
        if theoretical: task.insert(0, ('Theoretical', self.theoretical()))
        if state_dict: task.append(('After Training', self.effective(state_dict=state_dict)))
        # 开始绘制图像
        for i, (title, heatmap) in enumerate(task):
            plt.subplot(1, len(task), i + 1)
            plt.title(title)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(heatmap, vmin=0, vmax=1, **imshow_kw)

    def effective(self, state_dict=None):
        Activation.identity = True
        x = torch.rand([self.n_sample, *self.img_size], requires_grad=True)
        if state_dict: self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)(x.to(self.device))
        return self._backward(x)

    def theoretical(self):
        Activation.identity = False
        x = torch.ones([1, *self.img_size], requires_grad=True)
        # 原地替换替换参数
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            fill = {'weight': 1 + np.random.random() * 1e-2, 'bias': 0,
                    'running_mean': 0, 'running_var': 1}.get(key.split('.')[-1], None)
            if fill is not None: nn.init.constant_(value, fill)
        self.model.to(self.device)(x.to(self.device))
        return np.sqrt(self._backward(x))

    def replace(self, model):
        ''' 更换模型中的激活函数'''
        for key, m in model._modules.items():
            if hasattr(m, 'inplace'):
                new = Activation()
                model._modules[key] = new
                new.__dict__.update(m.__dict__)
            self.replace(m)

    def _backward(self, x):
        fmap, self._fmap = self._fmap, None
        i = tuple(x // 2 for x in fmap.shape[2:])
        fmap[..., i[0], i[1]].sum().backward()
        res = x.grad.abs().view(-1, *self.img_size[1:]).max(dim=0)[0].sqrt().data.numpy()
        return res / res.max()


if __name__ == '__main__':
    from model import YamlModel
    from pathlib import Path

    m = YamlModel(Path('../config/ResNet.yaml'))
    #
    # a = torch.rand(1, 3, 8, 8)
    # a.requires_grad = True
    # x = m(a)[0, 0, 3, 3]
    # x.backward()
    # print(a.grad)
    with ReceptiveField(m, tar_layer=-6, img_size=256) as r:
        plt.imshow(r.effective())
    plt.show()
