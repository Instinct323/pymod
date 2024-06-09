import copy

import torch
import torch.nn.functional as F
from torch import nn

from .model import is_parallel


class EmaModel:
    """ Mean teachers are better role models
        通过学生的 state_dict 保存 / 加载参数"""

    def __init__(self,
                 model: nn.Module,
                 bp_times: int = 2000,
                 decay: float = .999):
        assert isinstance(model, nn.Module)
        self.__model = model
        # 冻结 ema 的所有参数, 并将参数加入 model
        self.__ema = self.__model.ema = copy.deepcopy(self.__model)
        for p in self.__ema.parameters(): setattr(p, "requires_grad", False)
        # 记录 EMA 的次数
        self.__ema.register_buffer("ema_t", torch.tensor([0], dtype=torch.int64))
        self.decay = lambda: decay * (1 - torch.exp(-self.__ema.ema_t * 10 / bp_times).item())
        self.forward = self.__ema

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        self.forward.eval()
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def update(self):
        self.__ema.ema_t += 1
        d = self.decay()
        # Exponential moving average weight
        state_t, state_s = self.__ema.state_dict(), self.__model.state_dict()
        for k, v in state_t.items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * state_s[k]

    def mse(self, x, y):
        self.update()
        loss = F.mse_loss(self(x), y)
        return loss - loss.detach()

    def DP(self):
        self.forward = self.forward if is_parallel(self.forward) else nn.DataParallel(self.forward)
        return self
