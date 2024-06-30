import bisect

import matplotlib.pyplot as plt
import numpy as np


def runge_kutta(pdfunc, init, dt, n):
    """ :param pdfunc: 偏微分函数
        :param init: 初值条件"""
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
        """ :param s: 当前状态
            :param t: 转移次数
            :return: 新状态概率分布"""
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
        if isinstance(data, np.ndarray):
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


if __name__ == "__main__":
    np.set_printoptions(3, suppress=True)

    for i in range(10):
        T = np.random.rand(5, 5)
        T /= T.sum(axis=0)

        m = MarkovChain(T)
        s = m.steady_state()

        print(s)
        print(m.update_state(s, 10))
        print()
