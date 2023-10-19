import bisect

import matplotlib.pyplot as plt
import numpy as np


def runge_kutta(pdfunc, init, dt, n):
    ''' :param pdfunc: 偏微分函数
        :param init: 初值条件'''
    ret = [np.array(init)]
    for _ in range(n):
        k1 = pdfunc(ret[-1])
        k2 = pdfunc(ret[-1] + dt / 2 * k1)
        k3 = pdfunc(ret[-1] + dt / 2 * k2)
        k4 = pdfunc(ret[-1] + dt * k3)
        ret.append(ret[-1] + dt / 6 * (k1 + 2 * (k2 + k3) + k4))
    return np.stack(ret)


def laida_bound(data):
    ''' 拉以达边界
        :param data: 单指标向量'''
    std = np.std(data)
    mean = np.mean(data)
    return mean - std * 3, mean + std * 3


def adjusted_r_squared(pred, target, n_features):
    ''' 拟合优度: 校正决定系数 R^2'''
    n_samples = pred.size
    tss = np.square(target - target.mean()).sum()
    rss = np.square(pred - target).sum()
    adjusted = (n_samples - 1) / max([n_samples - n_features - 1, 1])
    return 1 - rss / tss * adjusted


class PolyFun(np.poly1d):
    w = property(lambda self: self.c[::-1])

    def gradf(self):
        return np.poly1d(self.c[:-1] * np.arange(self.o, 0, -1))


class PolyFit(PolyFun):
    ''' 多项式拟合'''

    def __init__(self, x, y, deg, plot=False):
        super().__init__(np.polyfit(x, y, deg))
        self.y = y
        self.pred = self(x)
        # 绘制拟合结果
        if plot:
            plt.plot(x, self.y, label='true', color='orange')
            plt.plot(x, self.pred, label='pred', color='deepskyblue', linestyle='--')
            plt.legend(), plt.show()

    def abs_error(self):
        return np.abs(self.pred - self.y)

    def rela_error(self):
        return self.abs_error() / self.y


class PCA:
    ''' 主成分分析
        :param x: 相关系数矩阵 (e.g., np.cov)
        :param thresh: 累积贡献率阈值
        :ivar contri: 各个主成分的贡献率
        :ivar rota: 旋转降维矩阵'''

    def __init__(self, x, thresh=0.90):
        assert 0. < thresh <= 1.
        lam, vec = np.linalg.eig(x)
        # 降序排序: 特征值、特征向量
        order = np.argsort(lam)[::-1]
        lam, vec = lam[order], vec[:, order]
        # 根据累积贡献率决定降维程度 (二分查找)
        lam /= lam.sum()
        i = bisect.bisect(np.cumsum(lam), thresh)
        self.contri = lam[:i]
        self.rota = vec[:, :i]

    def __call__(self, x):
        return x @ self.rota

    def __repr__(self):
        contri = tuple(map(lambda x: round(x, 3), self.contri))
        return f'{type(self).__name__}{contri}'
