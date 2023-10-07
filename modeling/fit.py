# from scipy.interpolate import interp1d, interp2d
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm import trange


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 调整曲线和散点的颜色
    colors = ['deepskyblue', 'orange']
    plt.rcParams['figure.figsize'] = [4.0, 3.0]
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    np.set_printoptions(precision=3, suppress=True)
    process = lambda x: np.stack([x, x ** 2], axis=1)

    # 实验数据
    data = [
        np.array([[0.2, 0.3, 0.4, 0.5, 0.6],
                  [945, 897, 834, 746, 666]])
    ]

    for i, (x, y) in enumerate(data):
        plt.title(f'S{"V" * i}PWM调速方式的机械特性')

        z = process(x)
        plt.scatter(x, y, c=colors[0], label='true')

        reg = LinearRegression().fit(z, y)
        # 对表二的单独操作
        if i == 1: pass

        x = np.linspace(x.min(), x.max(), 50)
        plt.plot(x, reg.predict(process(x)), c=colors[1], label='pred')
        # plt.legend(frameon=False)
        plt.show()
