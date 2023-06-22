from typing import Callable, List, Tuple

import numpy as np
from torch import nn
from torch.nn.utils import prune


class Pruner:
    target = tuple(getattr(nn, f'Conv{i}d') for i in range(1, 4)) + \
             tuple(getattr(nn, f'ConvTranspose{i}d') for i in range(1, 4)) + \
             (nn.Linear, nn.Bilinear)

    def __init__(self, model: nn.Module):
        self.model = model
        self.param = self.param_search(model, name='weight')

    def planning(self,
                 fitness: Callable[[nn.Module], float],
                 max_rate: float,
                 test_times: int = None) -> np.ndarray:
        ''' 使用不同比例裁剪网络, 给出性能对比'''
        rate = np.linspace(0, max_rate, test_times + 1) \
            if test_times else np.arange(0, max_rate + 0.01, 0.01)
        fitn = [fitness(self.model)]
        # 性能下降幅度大于 15% 退出
        for i in rate[1:]:
            fit = fitness(self.pruning(i))
            fitn.append(fit)
            if fit < .85 * fitn[0]: break
        return np.stack(tuple(zip(rate, fitn)))

    def pruning(self, rate) -> nn.Module:
        ''' 按照比例裁剪网络'''
        prune.global_unstructured(self.param, pruning_method=prune.L1Unstructured, amount=rate)
        return self.model

    def save(self):
        ''' 保存裁剪后的参数'''
        for m, name in self.param: prune.remove(m, name)

    def param_search(self, model: nn.Module, name: str) -> List[Tuple[nn.Module, str]]:
        ''' 搜索具有属性 name 的模块'''
        ret = []
        for m in model._modules.values():
            if isinstance(m, self.target): ret.append((m, name))
            ret += self.param_search(m, name)
        return ret
