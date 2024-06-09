from functools import partial
from typing import Sequence, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import SimpleDataset, to_tensor, ImagePool
from .utils import LOGGER, Path


class SoftLabel:

    @staticmethod
    def dump(x, topk=None):
        if not topk: return x.tolist()
        topk = torch.argsort(x, descending=True)[:topk].tolist()
        # 稀疏化存储, 其它类别置零
        x /= x[topk].sum()
        cache = {i: x[i].item() for i in topk}
        cache["size"] = len(x)
        return cache

    @staticmethod
    def load(x):
        if isinstance(x, dict):
            x, cache = torch.zeros(x.pop("size")), x
            for i, v in cache.items(): x[i] = v
            return x
        return torch.tensor(x)


class Knowledge(SimpleDataset):
    """ :param cache: 教师知识的存储文件
        :ivar writeable: 可写状态
        :ivar as_read: 保存当前知识库"""
    orient = "index"

    def __init__(self,
                 imgpool: ImagePool,
                 indexes: Sequence,
                 aughyp: dict,
                 cache: Path,
                 topk: int = None):
        super().__init__(imgpool=imgpool, indexes=indexes, aughyp=aughyp)
        self.cache = cache
        self.topk = topk
        assert self.aug, "Data augmentations component not found"
        if not cache.is_file():
            self.as_write()
        else:
            self.knowledge = cache.binary()
            self.as_read(save=False)

    def as_write(self):
        self.writeable = True
        if len(getattr(self, "knowledge", tuple())):
            LOGGER.info(f"Knowledge has been completely emptied")
        self.knowledge = pd.DataFrame(columns=list(self.aug.get_param().keys()) + ["__logits__"])

    def as_read(self, save=True):
        self.writeable = False
        if save:
            LOGGER.info(f"Save knowledge to {self.cache}")
            self.cache.binary(self.knowledge)
        self.n = len(self.knowledge) // len(self)

    def distill(self,
                soft_getter: Callable = partial(F.softmax, dim=-1)):
        """ :param soft_getter: 从 logits 中提取 softlabel 的函数"""
        assert not self.writeable, "Soft labels cannot be obtained in write mode"
        try:
            # 不可逆地将 logits 转为 softlabel
            self.knowledge["__soft__"] = [
                SoftLabel.dump(soft_getter(torch.tensor(logits)), topk=self.topk)
                for logits in tqdm(self.knowledge.pop("__logits__"), desc="logits -> softlabel")]
        except KeyError:
            LOGGER.warning("Knowledge cannot be redistilled")

    @torch.no_grad()
    def generate(self, logit_getter, batch_size, n):
        """ :param logit_getter: 从图像中提取 logits 的函数
            :param batch_size: 生成知识的批次大小
            :param n: 知识量相对于数据集的倍数"""
        assert self.writeable, "Cannot input knowledge in read mode"
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=False)
        for i in range(n):
            # 遍历 n 遍数据集, 搜集增强状态
            for img, hard, param in tqdm(loader, desc=f"Epoch {i + 1}/{n}"):
                param = pd.DataFrame(param)
                param["__logits__"] = logit_getter(img).cpu().tolist()
                self.knowledge = pd.concat((self.knowledge.convert_dtypes(), param), ignore_index=True)
        self.as_read()

    def __getitem__(self, item):
        img, hardlabel = self.imgpool[self.indexes[item]]
        # write 模式 + param
        if self.writeable:
            extra = param = self.aug.get_param()
            extra["__i__"] = item
        # read 模式 + 软标签
        else:
            i = np.random.randint(self.n) * len(self) + item
            param = self.knowledge.iloc[i].to_dict()
            # 校验图像 ID
            assert item == param.pop("__i__")
            extra = SoftLabel.load(param.pop("__soft__"))
        return to_tensor(self.aug.apply(img, param)), hardlabel, extra

    def __repr__(self):
        return str(self.knowledge)
