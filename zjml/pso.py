from typing import Sequence, Optional

import numpy as np
from tqdm import trange
import pandas as pd

DTYPE = np.float32
EPS = np.finfo(DTYPE).eps


class ParticleSwarmOpt:
    """ 粒子群优化器
        :param n: 粒子群规模
        :param lr: 学习率
        :param well_radio: 优粒子百分比
        :param best_unit: 已知最优个体"""

    def __init__(self,
                 n: int,
                 lr: float = 1.,
                 well_radio: float = 0.15,
                 best_unit: Optional[Sequence] = None):
        # 记录系统参数
        self._n = n
        self._well_size = max(2, round(n * well_radio))
        self._lr = lr
        # 记录最优个体
        if best_unit:
            self.best_unit = np.array(best_unit, dtype=DTYPE)
            self.best_fitness = self.fitness(self.best_unit[None])
        else:
            self.best_unit = None
            self.best_fitness = - np.inf
        # 记录不产生最优个体的次数
        self._angry = 0
        # 生成粒子群
        self.particle = self.generate(self._n)
        if isinstance(self.best_unit, np.ndarray): self.particle[0] = self.best_unit
        self.inertia = np.zeros_like(self.particle)
        self.log = []

    def generate(self, n: int) -> np.ndarray:
        """ 产生指定规模的群体"""
        raise NotImplementedError

    def fitness(self, particle: np.ndarray) -> np.ndarray:
        """ 适应度函数 (max -> best)"""
        raise NotImplementedError

    def revisal(self):
        """ 粒子修正 (e.g., 越界处理)"""
        return

    def fit(self, epochs: int,
            patience: int = np.inf,
            inertia_weight: float = 0.5,
            random_epochs_percent: float = 0.4,
            prefix="PSO-fit") -> float:
        """ :param epochs: 训练轮次
            :param patience: 允许搜索无进展的次数
            :param inertia_weight: 惯性权值
            :param random_epochs_percent: 随机搜索轮次百分比"""
        # 随机搜索轮次数
        random_epochs = int(random_epochs_percent * epochs)
        pbar = trange(epochs)
        for i in pbar:
            # 粒子互相影响下产生的移动量
            move_pace = self._move_pace()
            # 收敛检测
            if self._angry == patience: break
            # 根据轮次生成随机移动量, 随机缩放比例: [-1, 1]
            if i < random_epochs:
                self._angry = 0
                move_pace *= np.random.uniform(-1, 1, self.particle.shape)
            # 移动粒子: 吸引力 + 惯量 * 系数
            self.particle += (move_pace + inertia_weight * self.inertia) * self._lr
            self.inertia = move_pace
            # 群体补全
            need = self._n - len(self.particle)
            if need:
                self.particle = np.concatenate([self.particle, self.generate(need)], axis=0)
                self.inertia = np.concatenate([self.inertia, np.zeros([need, self.particle.shape[-1]])], axis=0)
            # 展示进度
            pbar.set_description((f"%-10s" + "%-10.4g") % (prefix, self.best_fitness))
        pbar.close()
        return self.best_unit, pd.DataFrame(self.log, columns=["fit-best", "fit-mean", "fit-std", "n-unique"])

    def _particle_slice(self, cond: np.ndarray) -> None:
        """ 粒子切片"""
        self.particle = self.particle[cond]
        self.inertia = self.inertia[cond]

    def _sort_unique(self):
        self.revisal()
        # 去除无限值
        fitness = np.array(self.fitness(self.particle), dtype=DTYPE)
        cond = np.isfinite(fitness)
        self._particle_slice(cond)
        fitness = fitness[cond]
        # 重叠检测
        order = np.argsort(fitness)[::-1]
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                if fitness[order[i]] == fitness[order[j]] and np.all(
                        self.particle[order[i]] == self.particle[order[j]]):
                    order[i] = -1
                    break
        # 应用排序结果
        order = order[order != -1]
        self._particle_slice(order)
        return fitness[order]

    def _fitness_factor(self) -> np.ndarray:
        """ 适应度因子"""
        fitness = self._sort_unique()
        # 更新全局最优的个体
        if fitness[0] > self.best_fitness:
            self._angry = 0
            self.best_fitness = fitness[0]
            self.best_unit = self.particle[0].copy()
        else:
            self._angry += 1
        self.log.append([fitness[0], fitness.mean(), fitness.std(), len(fitness)])
        # 只保留优粒子的适应度
        well_bound = np.sort(fitness)[- self._well_size]
        fitness = np.maximum(np.append(fitness, self.best_fitness) - well_bound, 0)
        fitness /= fitness.max() + EPS
        # 削弱最优适应度的影响力
        fitness[-1] = 1.1 - fitness[:-1].max()
        return fitness

    def _move_pace(self) -> np.ndarray:
        """ 根据 距离、适应度 产生的移动量"""
        fitness_factor = self._fitness_factor()
        # 粒子间的距离
        direct = np.append(self.particle, self.best_unit[None], axis=0) - self.particle[:, None]
        dist = np.square(direct).sum(axis=-1)
        # 归一化: 距离因子
        dist_max = dist.max(axis=1, keepdims=True)
        dist_factor = (dist_max - dist) / dist_max
        # 粒子间的影响力
        influence = fitness_factor * dist_factor
        return (direct * influence[..., None]).sum(axis=1)


class RangeOpt(ParticleSwarmOpt):
    """ 产生给定范围内的粒子群"""
    coord_range = None

    def generate(self, n: int) -> np.ndarray:
        """ 产生指定规模的群体"""
        x = np.random.random([n, len(self.coord_range)])
        return x * (self.coord_range[:, 1] - self.coord_range[:, 0]) + self.coord_range[:, 0]

    def revisal(self):
        """ 越界处理"""
        for j, (amin, amax) in enumerate(self.coord_range):
            self.particle[:, j] = np.clip(self.particle[:, j], a_min=amin, a_max=amax)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 定义 3 个自变量的范围
    COORD_RANGE = [0, 2], [2, 4], [4, 7]


    class My_PSO(RangeOpt):
        coord_range = np.array(COORD_RANGE, dtype=DTYPE)

        def fitness(self, particle):
            return np.sin(particle).sum(axis=-1)


    # 绘制正弦函数
    t = np.linspace(0, 7, 100)
    plt.plot(t, np.sin(t), color="deepskyblue")
    # 绘制自变量边界
    for bound in set(sum(COORD_RANGE, [])):
        plt.plot([bound] * 2, [-1, 1], color="aqua", linestyle="--")

    # 重写粒子群优化器, 并初始化
    pso = My_PSO(50)
    best, log = pso.fit(10)
    print(log)
    # 绘制最优解
    plt.scatter(best, np.sin(best), marker="p", c="orange")
    plt.show()
