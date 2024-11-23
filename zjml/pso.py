from typing import Sequence, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange

DTYPE = np.float32
EPS = 1e-8


class Scheduler:
    """ 优化参数调度器
        :param lr: 学习率
        :param w_self: 自身经验权重
        :param w_other: 群体经验权重
        :param w_inertia: 惯性权重
        :param t_exploration: 随机搜索的时间比例"""
    lr = property(lambda self: self.cur[0])
    w_self = property(lambda self: self.cur[1])
    w_other = property(lambda self: self.cur[2])
    w_inertia = property(lambda self: self.cur[3])
    w_exploration = property(lambda self: self.cur[4])

    def __init__(self,
                 lr: float = .2,
                 w_self: float = 2.,
                 w_other: float = 1.,
                 w_inertia: float = 0.5,
                 t_exploration: float = 0.5):
        self.org = np.array([lr, w_self, w_other, w_inertia, t_exploration], dtype=DTYPE)
        self.cur = self.org.copy()

    def step(self, progress: float):
        to_decay = [0, 1, 3]
        self.cur[to_decay] = self.org[to_decay] * ((1 - progress) * .9 + .1)
        self.cur[-1] = max(0, self.org[-1] - progress)
        return self


class ParticleSwarmOpt:
    """ 粒子群优化器
        :param n: 粒子群规模
        :param well_radio: 优粒子百分比
        :param best_unit: 已知最优个体"""

    def __init__(self,
                 n: int,
                 well_radio: float = 0.15,
                 best_unit: Optional[Sequence] = None):
        # 记录系统参数
        self._n = n
        self._well_size = max(2, round(n * well_radio))
        # 记录最优个体
        if best_unit:
            self.bestX = np.array(best_unit, dtype=DTYPE)
            self.bestY = self.fitness(self.bestX[None])
        else:
            self.bestX = None
            self.bestY = - np.inf
        # 记录不产生最优个体的次数
        self._angry = 0
        # 生成粒子群
        self.particle = self.generate(self._n)
        if isinstance(self.bestX, np.ndarray): self.particle[0] = self.bestX
        self.inertia = np.zeros_like(self.particle)
        self.fit_vec = self.fitness(self.particle).astype(DTYPE)
        self.loc_best = [np.zeros_like(self.particle), self.fit_vec.copy()]
        # 用于可视化
        self.log = pd.DataFrame(columns=["fit-best", "fit-std", "n-unique"])
        self.std_ema = None

    def generate(self, n: int) -> np.ndarray:
        """ 产生指定规模的群体"""
        raise NotImplementedError

    def fitness(self, particle: np.ndarray) -> np.ndarray:
        """ 适应度函数 (max -> best)"""
        raise NotImplementedError

    def revisal(self):
        """ 粒子修正 (e.g., 越界处理)"""
        return

    def fit(self,
            epochs: int,
            sche: Optional[Scheduler] = None,
            patience: int = np.inf,
            vis_itv: Union[float, int] = 0,
            prefix: str = "PSO-fit") -> float:
        """ :param epochs: 训练轮次
            :param sche: 优化参数调度器
            :param patience: 允许搜索无进展的次数
            :param vis_itv: 可视化的时间间隔 (比例 / 轮次)"""
        sche = sche or Scheduler()
        # 可视化间隔
        assert vis_itv >= 0
        vis_itv = max(1, int(vis_itv * epochs)) if isinstance(vis_itv, float) else vis_itv
        pbar = trange(epochs)
        for i in pbar:
            sche.step(i / epochs)
            # 收敛检测
            self._update()
            if vis_itv and i % vis_itv == 0: self.visualize()
            if self._angry == patience: break
            # 粒子互相影响下产生的移动量
            move_pace = (np.random.uniform(0, sche.w_self, [self._n, 1]) * self._motion_from_self()
                         + np.random.uniform(0, sche.w_other, [self._n, 1]) * self._motion_from_other())
            # 根据轮次生成随机移动量
            if sche.w_exploration:
                self._angry = 0
                bound = np.abs(move_pace).mean(axis=0) * sche.w_exploration
                move_pace += np.random.uniform(-bound, bound, self.particle.shape)
            # 移动粒子: 吸引力 + 惯量 * 系数
            self.particle += (move_pace + sche.w_inertia * self.inertia) * sche.lr
            self.inertia = move_pace
            # 展示进度
            pbar.set_description((f"%-10s" + "%-10.4g") % (prefix, self.bestY))
        pbar.close()
        return self.bestX, self.log

    def visualize(self):
        """ 可视化粒子群"""
        n = self.particle.shape[-1]
        # 根据适应度函数决定颜色, 大小
        particle = np.append(self.particle, self.bestX[None], axis=0)
        fmin = self.fit_vec.min()
        c = (np.append(self.fit_vec, self.bestY) - fmin) / (self.bestY - fmin + EPS)
        kwargs = dict(s=c * 80, c=c, cmap=plt.get_cmap("rainbow"), alpha=1)
        # 按照标准差决定上下限
        mean = self.bestX
        std = np.sqrt(np.square(particle - mean).mean(axis=0)) * 2 + EPS
        if self.std_ema is None:
            self.std_ema = std
        else:
            m = 0.1
            std = np.minimum(std, self.std_ema * 1.1)
            self.std_ema = m * std + (1 - m) * self.std_ema
        lb = mean - std
        ub = mean + std
        # 初始化画布
        plt.figure(323)
        plt.clf()
        fig = plt.subplot2grid((1, 3), (0, 0), colspan=2, **(dict(projection='3d') if n > 2 else {}))
        plt.title("particle")
        # fixme: 可视化粒子惯性, normal_i 归零时崩溃
        # normal_i = std.mean() * self.inertia / (np.linalg.norm(self.inertia, axis=-1).max() + EPS) / 20
        # fig.quiver(*self.particle[:, :n].T, *normal_i[:, :n].T, color="gray", alpha=0.5)
        # 可视化粒子位置
        plt.grid(True)
        if n == 1:
            plt.xlabel("x")
            plt.ylabel("fitness")
            plt.scatter(particle[:, 0], np.zeros_like(particle[:, 0]), **kwargs)
            plt.xlim(lb[0], ub[0])
        else:
            for i, a in enumerate("xyz"[:n]):
                getattr(fig, f"set_{a}label")(f"x{i + 1}")
                getattr(fig, f"set_{a}lim")(lb[i], ub[i])
            fig.scatter(*particle[:, :n].T, **kwargs)
        # 可视化适应度曲线
        plt.subplot(2, 3, 3)
        plt.title("fitness")
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.grid(True)
        plt.plot(self.log["fit-best"], color="deepskyblue")
        # 可视化有效粒子数目
        plt.subplot(2, 3, 6)
        plt.title("n-unique")
        plt.xlabel("iteration")
        plt.ylabel("n-unique")
        plt.grid(True)
        plt.plot(self.log["n-unique"], color="deepskyblue")
        # 结束绘制
        plt.tight_layout()
        plt.pause(1e-3)

    def _particle_slice(self, cond: np.ndarray) -> None:
        """ 粒子切片"""
        self.particle = self.particle[cond]
        self.inertia = self.inertia[cond]
        self.fit_vec = self.fit_vec[cond]
        self.loc_best = [x[cond] for x in self.loc_best]

    def _update(self):
        self.revisal()
        # 去除无限值
        self.fit_vec = self.fitness(self.particle).astype(DTYPE)
        self._particle_slice(np.isfinite(self.fit_vec))
        # 重叠检测
        self._particle_slice(np.argsort(self.fit_vec)[::-1])
        order = np.arange(len(self.fit_vec))
        for i in np.where(np.diff(self.fit_vec) > - EPS)[0]:
            if np.square(self.particle[i] - self.particle[i + 1]).sum() < EPS:
                order[i + 1] = -1
        self._particle_slice(order[order != -1])
        # 更新全局最优的个体
        if self.fit_vec[0] > self.bestY:
            self._angry = 0
            self.bestX = self.particle[0]
            self.bestY = self.fit_vec[0]
        else:
            self._angry += 1
        self.log.loc[len(self.log)] = [
            self.bestY, np.sqrt(np.square(self.fit_vec - self.bestY).mean()), len(self.fit_vec)]
        # 群体补全
        need = self._n - len(self.particle)
        if need:
            new = self.generate(need)
            fit = self.fitness(new)
            self.particle = np.concatenate([self.particle, new], axis=0)
            self.inertia = np.concatenate([self.inertia, np.zeros([need, self.particle.shape[-1]])], axis=0)
            self.fit_vec = np.concatenate([self.fit_vec, fit])
            self.loc_best[0] = np.concatenate([self.loc_best[0], new], axis=0)
            self.loc_best[1] = np.concatenate([self.loc_best[1], fit])

    def _motion_from_self(self) -> np.ndarray:
        better = np.sign(self.fit_vec - self.loc_best[1])
        direct = self.loc_best[0] - self.particle
        direct /= np.linalg.norm(direct, axis=-1, keepdims=True) + EPS
        # 更新个体最优
        idx = np.where(better > 0)[0]
        self.loc_best[0][idx] = self.particle[idx]
        self.loc_best[1][idx] = self.fit_vec[idx]
        return direct * better[:, None]

    def _motion_from_other(self) -> np.ndarray:
        # 适应度
        leader = np.arange(self._well_size)
        fitness = np.append(self.fit_vec[leader], self.bestY)
        fitness -= fitness[-2]
        fitness /= fitness[0] + EPS
        # 粒子间的距离
        leader = np.append(self.particle[leader], self.bestX[None], axis=0)
        direct = leader - self.particle[:, None]
        dist = np.sqrt(np.square(direct).sum(axis=-1))
        dist_max = dist.max(axis=1, keepdims=True)
        # 正向距离, 过近的忽略
        dist = (dist_max - dist) / dist_max
        dist[dist > 0.999] = 0
        # 粒子间的影响力
        influence = fitness * dist
        motion = (direct * influence[..., None]).sum(axis=1)
        return motion / np.linalg.norm(motion, axis=-1, keepdims=True)


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
    from utils import rosenbrock_func


    class My_PSO(RangeOpt):
        coord_range = np.array([[-0, 1]] * 2, dtype=DTYPE) * 2

        def fitness(self, particle):
            return - rosenbrock_func(particle)


    # 重写粒子群优化器, 并初始化
    pso = My_PSO(100)
    best, log = pso.fit(500, vis_itv=5)
    print(best)
    print(log)
    plt.show()
