from typing import Sequence, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange

DTYPE = np.float32
EPS = 1e-8


class Scheduler:
    """ 优化参数调度器
        :param dt: 单位时间
        :param w_self: 自身经验权重
        :param w_other: 群体经验权重
        :param w_inertia: 惯性权重"""
    dt = property(lambda self: self.cur[0])
    w_self = property(lambda self: self.cur[1])
    w_other = property(lambda self: self.cur[2])
    w_inertia = property(lambda self: self.cur[3])

    def __init__(self,
                 dt: float = .2,
                 w_self: float = 2.,
                 w_other: float = 2.,
                 w_inertia: float = 1.):
        self.org = np.array([dt, w_self, w_other, w_inertia], dtype=DTYPE)
        self.cur = self.org.copy()

    def step(self, progress: float):
        # modify `self.cur` here
        return self


class DecayScheduler(Scheduler):
    """ 优化参数调度器 (线性衰减)"""

    def step(self, progress: float):
        to_decay = [0]
        self.cur[to_decay] = self.org[to_decay] * ((1 - progress) * .9 + .1)
        return self


class ParticleSwarmOpt:
    """ 粒子群优化器 (reference: https://zhuanlan.zhihu.com/p/346355572)
        :param n: 粒子群规模
        :param elite_radio: 精英粒子百分比
        :param best_unit: 已知最优个体"""
    b_rm_dup = True
    fig_id = 323

    def __init__(self,
                 n: int,
                 elite_radio: float = 0.1,
                 best_unit: Optional[Sequence] = None):
        # 记录系统参数
        self._n = n
        self._elite_size = min(n - 1, max(2, round(n * elite_radio)))
        # 记录最优个体
        if best_unit:
            self.bestX = np.array(best_unit, dtype=DTYPE)
            self.bestY = self.fitness(self.bestX[None])[0]
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
        self.log = pd.DataFrame(columns=["fit-best", "fit-mean", "r-unique"])
        self.std_ema = None

    def generate(self, n: int) -> np.ndarray:
        """ 产生指定规模的群体"""
        raise NotImplementedError

    def fitness(self, particle: np.ndarray) -> np.ndarray:
        """ 适应度函数 (max -> best)"""
        raise NotImplementedError

    def revisal(self):
        """ 粒子修正 (e.g., 越界处理; 默认为空)"""
        return

    def fit(self,
            epochs: int,
            sche: Optional[Scheduler] = None,
            patience: int = np.inf,
            vis_itv: Union[float, int] = 0,
            plt_video: Optional["PltVideo"] = None):
        """ :param epochs: 训练轮次
            :param sche: 优化参数调度器
            :param patience: 允许搜索无进展的次数
            :param vis_itv: 可视化的时间间隔 (比例 / 轮次)"""
        prefix = "PSO-fit"
        sche = sche or Scheduler()
        # 可视化间隔
        assert vis_itv >= 0
        vis_itv = max(1, int(vis_itv * epochs)) if isinstance(vis_itv, float) else vis_itv
        pbar = trange(epochs)
        for i in pbar:
            sche.step(i / epochs)
            self._update()
            self.visualize(i, vis_itv, plt_video)
            # 收敛检测
            if self._angry == patience: break
            # 粒子互相影响下产生的速度
            v = (np.random.uniform(0, sche.w_self, [self._n, 1]) * self._vel_from_self()
                 + np.random.uniform(0, sche.w_other, [self._n, 1]) * self._vel_from_other())
            # 移动粒子: 吸引力 + 惯量 * 系数
            self.particle += (v + sche.w_inertia * self.inertia) * sche.dt
            self.inertia = v
            # 展示进度
            pbar.set_description((f"%-10s" + "%-10.4g") % (prefix, self.bestY))
        pbar.close()
        return self.bestX, self.log

    def visualize(self,
                  i: int,
                  vis_itv: int,
                  plt_video: ["PltVideo"] = None,
                  m: float = 0.02):
        """ 辅助函数: 可视化粒子群"""
        if not vis_itv or i % vis_itv: return
        n = min(3, self.particle.shape[-1])
        # 根据适应度函数决定颜色, 大小
        particle = np.append(self.particle, self.bestX[None], axis=0)
        fmin = self.fit_vec.min()
        c = (np.append(self.fit_vec, self.bestY) - fmin) / (self.bestY - fmin + EPS) * 0.9 + 0.1
        kwargs = dict(s=c * 80, c=c, cmap=plt.get_cmap("rainbow"), alpha=1)
        # 按照标准差决定上下限
        mean = self.bestX
        std = np.sqrt(np.square(particle - mean).mean(axis=0)) * 2 + EPS
        if self.std_ema is None:
            self.std_ema = std
        else:
            std = np.minimum(std, self.std_ema * 1.1)
            self.std_ema = m * std + (1 - m) * self.std_ema
        lb = mean - std
        ub = mean + std
        # 初始化画布
        plt.figure(self.fig_id)
        plt.clf()
        # 可视化粒子位置
        grids = plt.subplot2grid((1, 3), (0, 0), colspan=2, **(dict(projection='3d') if n > 2 else {}))
        plt.title(f"best-fit: {self.bestY:f}")
        plt.grid(True)
        if n == 1:
            plt.xlabel("x")
            plt.ylabel("fitness")
            plt.scatter(particle[:, 0], np.zeros_like(particle[:, 0]), **kwargs)
            plt.xlim(lb[0], ub[0])
        else:
            for i, a in enumerate("xyz"[:n]):
                getattr(grids, f"set_{a}label")(f"x{i + 1}")
                getattr(grids, f"set_{a}lim")(lb[i], ub[i])
            grids.scatter(*particle[:, :n].T, **kwargs)
        # 可视化适应度曲线
        plt.subplot(2, 3, 3)
        plt.ylabel("fitness")
        plt.grid(True)
        plt.plot(self.log["fit-best"], color="deepskyblue", label="best")
        # plt.plot(self.log["fit-mean"], color="orange", label="mean")
        # plt.legend()
        # 可视化有效粒子数目
        plt.subplot(2, 3, 6)
        plt.xlabel("iteration")
        plt.ylabel("r-unique")
        plt.ylim(0, 1.15)
        plt.grid(True)
        plt.plot(self.log["r-unique"], color="deepskyblue")
        plt.hlines(1, 0, len(self.log) - 1, color="gray", linestyle="--")
        plt.fill_between(np.arange(len(self.log)), self.log["r-unique"], color="deepskyblue", alpha=0.3)
        # 结束绘制
        plt.tight_layout()
        if plt_video: plt_video.write()
        plt.pause(1e-3)

    def _particle_slice(self, cond: np.ndarray) -> None:
        """ 辅助函数: 粒子切片"""
        self.particle = self.particle[cond]
        self.inertia = self.inertia[cond]
        self.fit_vec = self.fit_vec[cond]
        self.loc_best = [x[cond] for x in self.loc_best]

    def _unique_mask(self):
        order = np.arange(len(self.fit_vec))
        for i in np.where(np.diff(self.fit_vec) > - EPS)[0]:
            if np.linalg.norm(self.particle[i] - self.particle[i + 1]) < EPS:
                order[i + 1] = -1
        return order[order != -1]

    def _update(self):
        """ 主流程: 更新粒子群"""
        self.revisal()
        # 去除无限值
        self.fit_vec = self.fitness(self.particle).astype(DTYPE)
        self._particle_slice(np.isfinite(self.fit_vec))
        # 按照适应度降序排序
        self._particle_slice(np.argsort(self.fit_vec)[::-1])
        # 重叠检测
        order = self._unique_mask()
        if self.b_rm_dup: self._particle_slice(order)
        # 更新全局最优的个体
        if self.fit_vec[0] > self.bestY:
            self._angry = 0
            self.bestX = self.particle[0].copy()
            self.bestY = self.fit_vec[0]
        else:
            self._angry += 1
        self.log.loc[len(self.log)] = [self.bestY, self.fit_vec.mean(), len(order) / self._n]
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

    def _vel_from_self(self) -> np.ndarray:
        """ 主流程: 自身经验的影响"""
        better = np.sign(self.fit_vec - self.loc_best[1])
        direct = self.loc_best[0] - self.particle
        # direct /= np.linalg.norm(direct, axis=-1, keepdims=True) + EPS
        # 更新个体最优
        idx = np.where(better > 0)[0]
        self.loc_best[0][idx] = self.particle[idx]
        self.loc_best[1][idx] = self.fit_vec[idx]
        return direct * better[:, None]

    def _vel_from_other(self) -> np.ndarray:
        """ 主流程: 群体经验的影响"""
        # 适应度
        leader = np.arange(self._elite_size)
        fitness = np.append(self.fit_vec[leader], self.bestY)
        fitness = fitness - self.fit_vec[:, None]
        # 粒子间的距离
        leader = np.append(self.particle[leader], self.bestX[None], axis=0)
        direct = leader - self.particle[:, None]
        dist = np.sqrt(np.square(direct).sum(axis=-1))
        dist_max = dist.max(axis=1, keepdims=True)
        # 正向距离
        dist = (dist_max - dist) / dist_max
        # dist[dist > 0.999] = 0
        # 粒子间的影响力
        influence = fitness * dist
        return direct[np.arange(len(direct)), influence.argmax(axis=-1)]


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
    from utils import auckley_func

    plt.rcParams["figure.figsize"] = [9, 4.8]


    class My_PSO(RangeOpt):
        coord_range = np.array([[-1, 1]] * 2, dtype=DTYPE) * 10

        def fitness(self, particle):
            return - auckley_func(particle)

        @classmethod
        def main(cls):
            import cv2
            from pathlib import Path
            from pymod.zjplot.utils import PltVideo
            from pymod.utils.zjcv import VideoWriter

            root = Path(r"D:\Downloads")
            cls.b_rm_dup = False

            file = root / ("auck-" + ("best.mp4" if cls.b_rm_dup else "w-dup.mp4"))
            with PltVideo(323, VideoWriter(file, cvt_color=cv2.COLOR_RGB2BGR)) as pv:
                # 初始化 PSO, 开始优化
                best, log = My_PSO(100).fit(200, DecayScheduler(), vis_itv=1, plt_video=pv)
            print(best, log, sep="\n")
            plt.show()


    My_PSO.main()
