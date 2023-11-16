import copy
import logging
import warnings

import numpy as np
import pandas as pd
import yaml

from .result import Result

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class ModelScaling:
    ''' 模型复合缩放规划
        :param stride: 网络步长
        :param flops: 目标 FLOPS
        :param w_stride: 网络宽度增益的步长
        :param r_max: 分辨率的最高倍率'''
    orient = 'index'

    def __init__(self, project, cfg, stride, flops=2.,
                 d_keep=True, w_stride=.125, r_max=2):
        self.attr = 'depth_multiple', 'width_multiple', 'img_size'
        # 读取 yaml 文件并设置为属性
        self.project = project
        self.project.mkdir(parents=True, exist_ok=True)
        if not isinstance(cfg, dict):
            cfg = yaml.load(cfg.read_text(), Loader=yaml.Loader)
        self.cfg = cfg
        # 读取文件中的计划, 或者创建计划
        plan_file = project / 'scaling.json'
        if plan_file.is_file():
            self.plans = pd.read_json(plan_file, orient=self.orient)
        else:
            depth, width = (cfg.setdefault(k, 1.) for k in self.attr[:2])
            img_size = cfg[self.attr[2]]
            # 获取搜索空间
            r_stride = stride / img_size
            r = np.arange(np.round_(.5 / r_stride) * r_stride, r_max + 0.01, r_stride)
            d = np.array([cfg['depth_multiple']]) if d_keep else np.arange(1, flops + 1.01)
            d, r = map(lambda x: x.flatten(), np.meshgrid(d, r))
            w = np.sqrt(flops / (d / depth * r ** 2))
            w = np.round_(w * width / w_stride) * w_stride
            # 计算相似度
            ret = np.stack((d, w, r * img_size), axis=-1)
            scale = ret / np.array((depth, width, img_size))
            flops = np.round_(scale[:, 0] * np.square(scale[:, 1]) * np.square(scale[:, 2]), 3)
            similarity = np.square(scale - 1).sum(axis=-1)
            self.plans = pd.DataFrame(np.concatenate((ret, flops[:, None]), axis=1)[np.argsort(similarity)],
                                      columns=self.attr + ('flops',))
        self.plans.index = range(1, 1 + len(self.plans))
        self.plans.to_json(plan_file, orient=self.orient, indent=4)

    def __call__(self, fitness, epochs):
        ''' :param fitness(cfg, epoch) -> float: 适应度函数
            :param epochs: 复合缩放总轮次'''
        result = Result(self.project, title=self.attr + ('fitness',))
        # 检查训练总轮次
        if len(self.plans) < epochs:
            epochs = len(self.plans)
            warnings.warn(f'Limited by the number of schemes, epochs have been modified to {epochs}')
        # 输出训练计划
        LOGGER.info('\n' + str(self.plans[:epochs]))
        for epoch in range(len(result), epochs):
            d, w, r, f = self.plans.iloc[epoch]
            LOGGER.info(f'\n{__class__.__name__} epoch {epoch + 1}: d={d}  w={w}  r={r:.0f}')
            # 修改配置文件
            cfg = copy.deepcopy(self.cfg)
            cfg.update(self.plans.iloc[epoch].to_dict())
            cfg[self.attr[2]] = round(cfg[self.attr[2]])
            # 计算适应度
            fit = fitness(cfg, epoch + 1)
            result.record(tuple(cfg[k] for k in self.attr) + (fit,), epoch)
            LOGGER.info(('\n' + ' %9s' * 5) % ('scaling', 'depth', 'width', 'img_size', 'fitness'))
            LOGGER.info((' %9s' + ' %9.4g' * 4) % (f'{epoch + 1}/{epochs}', d, w, r, fit))
        best = result['fitness'].to_numpy().argmax() + 1
        LOGGER.info(f'\nModel scaling search is complete, the best epoch is {best}.\n')


if __name__ == '__main__':
    from pathlib import Path

    cfg = {'depth_multiple': 1, 'width_multiple': 1.25, 'img_size': 512}
    ms = ModelScaling(Path('__pycache__'), cfg, 32, flops=1.2)
    ms(lambda x, epoch: np.random.random(), 100)
