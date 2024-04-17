import collections
import copy
import warnings
from typing import Union, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn

to_2tuple = lambda x: x if isinstance(x, collections.abc.Iterable) and not isinstance(x, str) else (x,) * 2


class ReceptiveField:
    """ :param model: 需要进行可视化的模型
        :param tar_layer: 感兴趣的层, 其所输出特征图需有 4 个维度 [B, C, H, W]
        :param img_size: 测试时使用的图像尺寸
        :cvar n_sample: 生成的随机图像的数量, 详见 effective 方法"""
    n_sample = 8

    def __init__(self,
                 model: nn.Module,
                 tar_layer: Union[int, nn.Module],
                 img_size: Union[int, Tuple[int, int]],
                 in_channels: int = 3,
                 use_cuda: bool = False,
                 use_copy: bool = False):
        # 注册前向传播的挂钩
        self._fmap = None
        tar_layer = model[tar_layer] if isinstance(tar_layer, int) else tar_layer
        tar_layer.register_forward_hook(
            lambda module, x, y: setattr(self, "_fmap", y)
        )
        # 验证 tar_layer 的输出为特征图
        self.img_size = (in_channels,) + to_2tuple(img_size)
        self.model = model.eval()
        self.model(torch.zeros([1, *self.img_size]))
        assert self._fmap.dim() == 4, f"Invalid selection of tar_layer {type(tar_layer)}"
        # 对模型进行深拷贝
        if use_copy:
            try:
                self.model = copy.deepcopy(model).eval()
            except:
                warnings.warn("Fail to deep copy the model, the model will be modified in place")
        # 原地替换网络层
        self._replace(self.model)
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def compare(self, theoretical=True, original=True, state_dict=None, **imshow_kw):
        """ :param theoretical: 是否绘制理论感受野
            :param original: 是否绘制训练前的感受野
            :param state_dict: 模型权值, 如果提供则绘制训练后的感受野"""
        task = []
        if theoretical: task.append(("Theoretical", self.theoretical(light=0.3)))
        if original: task.append(("Before Training", self.effective()))
        if state_dict: task.append(("After Training", self.effective(state_dict=state_dict)))
        # 开始绘制图像
        for i, (title, heatmap) in enumerate(task):
            plt.subplot(1, len(task), i + 1)
            plt.title(title)
            plt.xticks([]), plt.yticks([])
            plt.imshow(heatmap, vmin=0, vmax=1, **imshow_kw)

    def effective(self, state_dict=None):
        """ :param state_dict: 模型权值, 如果提供则绘制训练后的感受野"""
        x = torch.rand([self.n_sample, *self.img_size], requires_grad=True)
        # 加载模型参数
        if state_dict: self.model.load_state_dict(state_dict, strict=False)
        return self._backward(x)

    def theoretical(self, light=1.):
        """ :param light: 理论感受野的亮度 [0, 1]"""
        x = torch.ones([1, *self.img_size], requires_grad=True)
        return (self._backward(x) > 0) * light

    def _replace(self, model):
        for key, m in model._modules.items():
            # 更换 MaxPool
            if isinstance(m, nn.MaxPool1d):
                model._modules[key] = nn.AvgPool1d(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding)
            elif isinstance(m, nn.MaxPool2d):
                model._modules[key] = nn.AvgPool2d(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding)
            self._replace(m)

    def _backward(self, x):
        self.model.to(self.device)(x.to(self.device))
        fmap, self._fmap = self._fmap, None
        # 获取特征图中心点的坐标, 并反向传播该点的梯度
        i = tuple(x // 2 for x in fmap.shape[2:])
        fmap[..., i[0], i[1]].sum().backward()
        # 取出原图像的梯度, 并对 B C 维度求取最大值
        res = x.grad.abs().view(-1, *self.img_size[1:]).max(dim=0)[0].sqrt().cpu().data.numpy()
        return res / (res.max() + 1e-8)


if __name__ == "__main__":
    from torchvision.models import resnet18

    # Step 1: 刚完成初始化的模型, 权重<完全随机>, 表 "训练前"
    m = resnet18()
    print(m)
    m.layer3[1].conv1.dilation = 2
    m.layer3[1].conv1.padding = 2

    # Step 2: 训练完成后的 state_dict, 等待 ReceptiveField 加载
    state_dict = resnet18(pretrained=True).state_dict()

    # Step 3: 绘制感受野 (设置 ReceptiveField 的 use_copy=True, 将创建模型的深拷贝副本)
    with ReceptiveField(m, tar_layer=m.layer3, img_size=256, use_copy=True) as r:
        r.compare(state_dict=state_dict)
        # 理论感受野的尺寸
        s = round(r.theoretical().sum() ** 0.5)
        print(f"Theoretical RF: {s}×{s}")
    plt.show()

    # Step 4: 加载模型的参数
    m.load_state_dict(state_dict)
