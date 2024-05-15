import random
from collections import deque
from pathlib import Path
from typing import Union

from torch import nn

from .trainer import Trainer


class ReplayBuffer:

    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def add(self, *args):
        # state, action, reward, next_state, done
        self.buffer.append(args)

    def __bool__(self):
        return len(self.buffer) > self.buffer.maxlen // 2

    def __len__(self):
        return len(self.buffer) // self.batch_size

    def __iter__(self):
        def generator():
            for i in range(len(self)):
                yield zip(*random.sample(self.buffer, self.batch_size))

        return generator()


class DQN(Trainer):
    """ Deep Q Network
        :param qnet: 动作值函数网络
        :param buffer_cap: 经验回放池容量"""

    def __init__(self,
                 qnet: nn.Module,
                 project: Path,
                 hyp: Union[Path, dict],
                 buffer_cap: int):
        self.qnet, self.qnet_target = qnet, qnet
        super().__init__(qnet, project, hyp)
        self.update_qnet_target()
        # 经验回放池
        self.buffer = ReplayBuffer(buffer_cap, self.hyp.get("batch_size"))
        self.load_ckpt()

    def update_qnet_target(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def load_ckpt(self, file: str = "last.pt") -> dict:
        ckpt = super().load_ckpt(file)
        if ckpt:
            self.qnet_target.load_state_dict(ckpt.get("model_tag"))
            self.buffer = ckpt.get("replay_buffer")
        return ckpt

    def save_ckpt(self, files: list = ["last.pt"], **ckpt_kwd):
        ckpt_kwd.update({
            "model_tag": self.qnet_target.state_dict(),
            "replay_buffer": self.buffer
        })
        super().save_ckpt(files, **ckpt_kwd)
