import copy

import torch
import torch.nn.functional as F
from torch import nn


class EmaModel:
    ''' Mean teachers are better role models
        通过学生的 state_dict 保存 / 加载参数'''
    teacher = property(fget=lambda self: self.student.teacher)

    def __init__(self,
                 student: nn.Module,
                 bp_times: int = 2000,
                 decay: float = .999):
        self.student = student
        # 冻结 teacher 的所有参数, 并将参数加入 student
        student.teacher = copy.deepcopy(student).eval()
        tuple(setattr(p, 'requires_grad', False) for p in self.teacher.parameters())
        # 记录 EMA 的次数
        self.teacher.register_buffer('ema_t', torch.tensor([0], dtype=torch.int64))
        self.decay = lambda: decay * (1 - torch.exp(-self.teacher.ema_t * 10 / bp_times).item())

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.teacher(*args, **kwargs)

    @torch.no_grad()
    def update(self):
        self.teacher.ema_t += 1
        d = self.decay()
        # Exponential moving average weight
        state_t, state_s = self.teacher.state_dict(), self.student.state_dict()
        for k, v in state_t.items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * state_s[k].detach()

    def mse(self, x, y):
        self.update()
        loss = F.mse_loss(self(x), y)
        return loss - loss.detach()
