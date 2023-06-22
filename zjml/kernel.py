import functools

import matplotlib.pyplot as plt
import numpy as np


def kernel_func(x1, x2=None, kernel='rbf', args=(1,)):
    # K(x, x') = Φ(x)^T Φ(x')
    x2 = x2 if isinstance(x2, np.ndarray) else x1
    # linear function
    if kernel == 'linear':
        return x1 @ x2.T
    # polynomial function
    elif kernel == 'poly':
        degree, zeta, gamma = args + (1,) * (3 - len(args))
        inpro = x1 @ x2.T
        return np.power(zeta + inpro * gamma, degree)
    # radial basis function
    elif kernel == 'rbf':
        sigma = args[0]
        dist = np.square(x1[:, None] - x2).sum(axis=-1)
        return np.exp(- dist / (2 * sigma ** 2))
    raise ValueError(f'Unknown kernel function {kernel}')


class KernelRegression:

    def __init__(self, x, y, kernel='rbf', args=(1,)):
        self.y = y
        self.kf = functools.partial(kernel_func, x2=x, kernel=kernel, args=args)

    def predict(self, x):
        k = self.kf(x)
        w = k / k.sum(axis=-1, keepdims=True)
        return (w * self.y).sum(axis=-1)


if __name__ == '__main__':
    np.set_printoptions(3, suppress=True)

    x = (np.random.random([100, 1]) - 0.5) * 10
    y = np.sin(x).flatten()

    kr = KernelRegression(x, y, args=(0.5,))
    x_ = np.linspace(x.min(), x.max(), 100)
    fx = kr.predict(x_[:, None])

    plt.scatter(x.flatten(), y)
    plt.plot(x_, fx)
    plt.show()
