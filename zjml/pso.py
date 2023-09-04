from typing import Sequence, Optional

import numpy as np
from tqdm import trange

DTYPE = np.float32
INF = np.finfo(DTYPE).max
EPS = np.finfo(DTYPE).eps


class ParticleSwarmOpt:
    ''' 粒子群优化器
        n: 粒子群规模
        coord_info: 每个坐标的信息 (e.g.,取值范围)
        well_percent: 优粒子百分比
        lr: 学习率
        best_unit: 已知最优个体'''

    def __init__(self, n: int,
                 coord_info: Sequence,
                 well_percent: float = 0.15,
                 lr: float = 1.,
                 best_unit: Optional[Sequence] = None):
        # 记录系统参数
        self.coord_info = np.array(coord_info, dtype=DTYPE)
        self._n = n
        self._well_size = max(2, round(n * well_percent))
        self._lr = lr
        # 记录最优个体
        if best_unit:
            self.best_unit = np.array(best_unit, DTYPE)
            self.best_fitness = self.fitness(self.best_unit[None])
        else:
            self.best_unit = None
            self.best_fitness = - np.inf
        # 记录不产生最优个体的次数
        self._angry = 0

    def generate(self, n: int) -> np.ndarray:
        ''' 产生指定规模的群体'''
        x = np.random.random([n, len(self.coord_info)])
        return x * (self.coord_info[:, 1] - self.coord_info[:, 0]) + self.coord_info[:, 0]

    def revisal(self):
        ''' 越界处理'''
        for j, (amin, amax) in enumerate(self.coord_info):
            self.particle[:, j] = np.clip(self.particle[:, j], a_min=amin, a_max=amax)

    def fitness(self, particle: np.ndarray) -> np.ndarray:
        ''' 适应度函数 (max -> best)'''
        raise NotImplementedError

    def fit(self, epochs: int,
            patience: int = -1,
            inertia_weight: float = 0.5,
            random_epochs_percent: float = 0.2,
            prefix='PSO_fit') -> float:
        ''' epochs: 训练轮次
            patience: 允许搜索无进展的次数
            inertia_weight: 惯性权值
            random_epochs_percent: 随机搜索轮次百分比'''
        # 生成粒子群
        self.particle = self.generate(self._n)
        if isinstance(self.best_unit, np.ndarray): self.particle[0] = self.best_unit
        self.inertia = np.zeros_like(self.particle)
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
                move_pace *= 2 * np.random.random(self.particle.shape) - 1
            # 移动粒子: 吸引力 + 惯量 * 系数
            self.particle += (move_pace + inertia_weight * self.inertia) * self._lr
            self.inertia = move_pace
            # 粒子修正, 重叠检测
            self.revisal()
            unique = np.unique(self.particle, return_index=True, axis=0)[1]
            self._particle_filter(unique)
            # 群体补全
            need = self._n - len(self.particle)
            if need:
                self.particle = np.concatenate([self.particle, self.generate(need)], axis=0)
                self.inertia = np.concatenate([self.inertia, np.zeros([need, len(self.coord_info)])])
            # 展示进度
            pbar.set_description((f'%-10s' + '%-10.4g') % (prefix, self.best_fitness))
        pbar.close()
        return self.best_unit

    def _particle_filter(self, cond: np.ndarray) -> None:
        ''' 粒子筛选'''
        self.particle = self.particle[cond]
        self.inertia = self.inertia[cond]

    def _fitness_factor(self) -> np.ndarray:
        ''' 适应度因子'''
        fitness = np.array(self.fitness(self.particle), dtype=DTYPE)
        fitness[~ np.isfinite(fitness)] = - INF
        # 局部最优的个体
        cur_best_index = fitness.argmax()
        cur_best_fitness = fitness[cur_best_index]
        # 更新全局最优的个体
        if cur_best_fitness > self.best_fitness:
            self._angry = 0
            self.best_fitness = cur_best_fitness
            self.best_unit = self.particle[cur_best_index].copy()
        else:
            self._angry += 1
        # 只保留优粒子的适应度
        well_bound = np.sort(fitness)[- self._well_size]
        fitness = np.maximum(np.append(fitness, self.best_fitness) - well_bound, 0)
        fitness /= fitness.max() + EPS
        # 削弱最优适应度的影响力
        fitness[-1] = 1. - fitness[:-1].max()
        return fitness

    def _move_pace(self) -> np.ndarray:
        ''' 根据 距离、适应度 产生的移动量'''
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 定义 3 个自变量的范围
    COORD_RANGE = [0, 2], [2, 4], [4, 7]

    # 绘制正弦函数
    t = np.linspace(0, 7, 100)
    plt.plot(t, np.sin(t), color='deepskyblue')
    # 绘制自变量边界
    for bound in set(sum(COORD_RANGE, [])):
        plt.plot([bound] * 2, [-1, 1], color='aqua', linestyle='--')


    class My_PSO(ParticleSwarmOpt):

        def fitness(self, particle):
            return np.sin(particle).sum(axis=-1) - 1000


    # 重写粒子群优化器, 并初始化
    pso = My_PSO(100, coord_info=COORD_RANGE)
    best = pso.fit(10)
    print(best)
    # 绘制最优解
    plt.scatter(best, np.sin(best), marker='p', c='orange')
    plt.show()
