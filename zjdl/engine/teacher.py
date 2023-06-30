import logging
from functools import partial
from typing import Union, Sequence, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import SimpleDataset, to_tensor, ImagePool
from .utils import Path, LOGGER

SOFTMAX = partial(F.softmax, dim=-1)


class SoftLabel:

    @staticmethod
    def dump(x, topk=None):
        if not topk: return x.tolist()
        topk = torch.argsort(x, descending=True)[:topk].tolist()
        # 稀疏化存储, 其它类别置零
        x /= x[topk].sum()
        cache = {i: x[i].item() for i in topk}
        cache['size'] = len(x)
        return cache

    @staticmethod
    def load(x):
        if isinstance(x, dict):
            x, cache = torch.zeros(x.pop('size')), x
            for i, v in cache.items(): x[i] = v
            return x
        return torch.tensor(x)


class Knowledge(SimpleDataset):
    ''' attr:
            cache: 教师知识的存储文件
            training: 训练模式用于录入知识, 验证模式用于输出知识
        method:
            eval: 保存当前知识库
            append: (图像索引, 增强状态, 软标签)'''
    orient = 'index'

    def __init__(self,
                 imgpool: ImagePool,
                 indexes: Sequence,
                 hyp: Union[dict, Path],
                 cache: Path,
                 topk: int = None):
        super().__init__(imgpool=imgpool, indexes=indexes, aughyp=hyp)
        self.cache = cache
        self.topk = topk
        assert self.tf, 'Data augmentations component not found'
        # 知识库, 当前模式
        self.training = False
        if cache.is_file():
            self.knowledge = pd.read_json(cache, orient=self.orient)
            self.eval(save=False)
        else:
            self.train()

    def train(self):
        self.training = True
        if len(getattr(self, 'knowledge', tuple())):
            LOGGER.info(f'Knowledge has been completely emptied')
        self.knowledge = pd.DataFrame(columns=list(self.tf.get_param().keys()) + ['__logits__'])

    @torch.no_grad()
    def generate(self, model, batch_size, n):
        assert self.training, 'Cannot input knowledge in eval mode'
        device = model.device
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=False)
        for i in range(n):
            # 遍历 n 遍数据集, 搜集增强状态
            for img, hard, param in tqdm(loader, desc=f'Epoch {i + 1}/{n}'):
                param = pd.DataFrame(param)
                param['__logits__'] = model(img.to(device)).cpu().tolist()
                self.knowledge = pd.concat((self.knowledge.convert_dtypes(), param), ignore_index=True)
        self.eval()

    def eval(self, save=True):
        self.training = False
        if save:
            LOGGER.info(f'Save knowledge to {self.cache}')
            self.knowledge.to_json(self.cache, orient=self.orient, indent=4)
        self.n = len(self.knowledge) // len(self)

    def distill(self,
                func: Callable = SOFTMAX):
        assert not self.training, 'Soft labels cannot be obtained in training mode'
        try:
            soft = []
            for logits in tqdm(self.knowledge.pop('__logits__'), desc='logits -> softlabel'):
                soft.append(SoftLabel.dump(func(torch.tensor(logits)), topk=self.topk))
            # 不可逆地将 logits 转为 softlabel
            self.knowledge['__soft__'] = soft
        except KeyError:
            LOGGER.warning('Knowledge cannot be redistilled')

    def __getitem__(self, item):
        img, hardlabel = self.imgpool[self.indexes[item]]
        # 训练模式: + param
        if self.training:
            extra = param = self.tf.get_param()
            extra['__i__'] = item
        # 验证模式: + 软标签
        else:
            i = np.random.randint(self.n) * len(self) + item
            param = self.knowledge.iloc[i].to_dict()
            # 校验图像 ID
            assert item == param.pop('__i__')
            extra = SoftLabel.load(param.pop('__soft__'))
        return to_tensor(self.tf.apply(img, param)), hardlabel, extra
