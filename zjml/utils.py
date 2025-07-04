import bisect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def farthest_point_sampling(pts: np.ndarray,
                            num_samples: int | float):
    """
    Farthest Point Sampling (FPS) algorithm.
    :param pts: The input point cloud.
    :param num_samples: The number of points to sample.
    :return: The sampled points and their indices.
    """
    n = len(pts)
    if isinstance(num_samples, float): num_samples = int(num_samples * n)
    if num_samples > n: return pts, None

    distances = np.full(n, np.inf)
    indices = np.zeros(num_samples, dtype=int)
    indices[0] = np.random.randint(0, n)

    for i in range(num_samples - 1):
        distances = np.minimum(distances, np.linalg.norm(pts - pts[indices[i]], axis=1))
        indices[i + 1] = np.argmax(distances)

    return pts[indices], indices


def pearson(x, eps=1e-8):
    """
    皮尔逊相关系数
    :param x: [n, 2]
    """
    assert x.shape[1] == 2 and x.ndim == 2
    mean = x.mean(axis=0)
    unbiased = x - mean
    cov = np.mean(unbiased[:, 0] * unbiased[:, 1])
    sigma = np.sqrt(np.square(x).mean(axis=0) - np.square(mean))
    return cov / (np.prod(sigma) + eps)


def grading_mask(x: pd.Series, bounds: list, float_fmt="%.2f"):
    """ 分级掩码"""
    labels = ([f"<{float_fmt}" % bounds[0]] +
              [(f"{float_fmt}~{float_fmt}" % (bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)] +
              [f">{float_fmt}" % bounds[-1]])
    bounds = [-np.inf] + list(bounds) + [np.inf]
    mask = np.stack([(bounds[i] <= x) & (x < bounds[i + 1]) for i in range(len(bounds) - 1)], axis=1)
    return pd.DataFrame(mask, columns=labels, index=x.index)


def rosenbrock_func(x, a=1, b=100):
    """ :param x: [-2, 2]"""
    assert x.shape[-1] > 1

    return (np.square(a - x[..., :-1]).sum(axis=-1) +
            b * np.square(x[..., 1:] - np.square(x[..., :-1])).sum(axis=-1))


def auckley_func(x, a1=20, a2=0.2, a3=np.pi * 2):
    """ :param x: [-10, 10]"""
    it1 = np.sqrt(np.square(x).mean(axis=-1))
    it2 = np.cos(a3 * x).mean(axis=-1)
    return -a1 * np.exp(-a2 * it1) - np.exp(it2) + a1 + np.e


def runge_kutta(pdfunc, init, dt, n):
    """
    :param pdfunc: 偏微分函数
    :param init: 初值条件
    """
    ret = [np.array(init)]
    for _ in range(n):
        k1 = pdfunc(ret[-1])
        k2 = pdfunc(ret[-1] + dt / 2 * k1)
        k3 = pdfunc(ret[-1] + dt / 2 * k2)
        k4 = pdfunc(ret[-1] + dt * k3)
        ret.append(ret[-1] + dt / 6 * (k1 + 2 * (k2 + k3) + k4))
    return np.stack(ret)


def pauta_criterion(data):
    """ 拉依达边界
        :param data: 单指标向量"""
    std = np.std(data)
    mean = np.mean(data)
    return mean - std * 3, mean + std * 3


def adjusted_r_squared(pred, target, n_features):
    """ :return: 校正决定系数 R^2"""
    n_samples = pred.size
    tss = np.square(target - target.mean()).sum()
    rss = np.square(pred - target).sum()
    adjusted = (n_samples - 1) / max(n_samples - n_features - 1, 1)
    return 1 - rss / tss * adjusted


class MarkovChain:
    """ 马尔科夫链
        :param T: 转移概率矩阵"""

    def __init__(self, T, check=True):
        if check:
            assert np.all(T.shape == (len(T),) * 2), "Transitional matrix should be square"
            assert np.abs(T.sum(axis=0) - 1).sum() < 1e-6, "Sum of each row should be 1"
        self.T = T

    def steady_state(self):
        """ :return: 马氏链稳态概率分布"""
        u, s, vt = np.linalg.svd(self.T - np.eye(len(self.T)))
        return vt[-1] / vt[-1].sum()

    def update_state(self, s, t):
        """
        :param s: 当前状态
        :param t: 转移次数
        :return: 新状态概率分布
        """
        return np.linalg.matrix_power(self.T, t) @ s


class PolyFun(np.poly1d):
    w = property(lambda self: self.c[::-1])

    def gradf(self):
        return np.poly1d(self.c[:-1] * np.arange(self.o, 0, -1))


class PolyFit(PolyFun):
    """ 多项式拟合"""

    def __init__(self, x, y, deg, plot=False):
        super().__init__(np.polyfit(x, y, deg))
        self.y = y
        self.pred = self(x)
        # 绘制拟合结果
        if plot:
            plt.plot(x, self.y, label="true", color="orange")
            plt.plot(x, self.pred, label="pred", color="deepskyblue", linestyle="--")
            plt.legend(), plt.show()

    def abs_error(self):
        return np.abs(self.pred - self.y)

    def rela_error(self):
        return self.abs_error() / self.y


class PCA:
    """ 主成分分析
        :param x: 相关系数矩阵 (e.g., np.cov)
        :param thresh: 累积贡献率阈值
        :ivar contri: 各个主成分的贡献率
        :ivar rota: 旋转降维矩阵"""

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
        return f"{__class__.__name__}{contri}"


class HexagonalMesh:
    """ 六边形网格"""
    default = np.array(-1, dtype=np.int32)

    def __init__(self, w, h, data=None):
        self._w, self._h = w, h
        self.data = self.default[None, None].repeat(h, 0).repeat(w, 1)
        # 对给定的数据进行裁剪
        if data is not None:
            data = data.flatten()[:self.data.size].astype(self.default.dtype)
            self.data = data.reshape(self.data.shape)

    def neighbor(self):
        """ :returns left, right, top-left, top-right, bottom-left, bottom-right"""
        padh = self.default[None, None].repeat(self._h, 0)
        padv = self.default[None, None].repeat(self._w + 1, 1)
        # 水平相邻
        query = np.concatenate([padh, self.data, padh], 1)
        reth = np.stack([query[:, :-2], query[:, 2:]])
        # 竖直相邻
        query = np.stack([query[i, is_even: is_even + self._w + 1]
                          for i, is_even in enumerate(map((1).__and__, range(self._h)))])
        query = np.concatenate([padv, query, padv], 0)
        retv = np.stack([query[:-2, :-1], query[:-2, 1:], query[2:, :-1], query[2:, 1:]])
        return np.concatenate([reth, retv], 0)

    def pos(self):
        xb, yb = np.meshgrid(*map(np.arange, (self._w, self._h)))
        x = xb + 0.5 * (yb & 1 ^ 1)
        y = yb * np.sqrt(3) / 2
        return np.stack([x, y], -1)

    def __repr__(self):
        stream, length = [], 4
        # 生成字符串流
        for row in self.data:
            stream.append(list(map(str, row)))
            length = max(length, max(map(len, stream[-1])))
        # 格式化字符串流
        sep = " " * (length // 2)
        indent = " " * int(0.75 * length)
        for i, row in enumerate(stream):
            stream[i] = indent * (i & 1 ^ 1) + sep.join(map(lambda x: x.center(length), row)).rstrip()
        return "\n".join(stream)


def poly_interp(x: float,
                xs: np.ndarray,
                ys: np.ndarray):
    """ 拉格朗日插值
        :param x: 插值点
        :param xs: 插值节点
        :param ys: 插值值
        :return: 基函数, 插值结果"""
    basic_fun = ((x - xs) / (xs[:, None] - xs))
    basic_fun[np.arange(len(xs)), np.arange(len(xs))] = 1
    basic_fun = basic_fun.prod(axis=1)
    return basic_fun, (basic_fun * ys).sum().item()


if __name__ == "__main__":
    res = poly_interp(0.54, np.array([0.5, 0.6, 0.7]), np.array([-0.693147, -0.510826, -0.356675]))
    print(res)
