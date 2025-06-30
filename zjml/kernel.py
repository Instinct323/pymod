from functools import partial

import matplotlib.pyplot as plt
import numpy as np


def kernel_select(array, k, s=1, d=1, pad_value=0, axis=0):
    """
    :param array: NumPy array to be processed
    :param k: kernel size
    :param s: stride
    :param d: dilation
    :param pad_value: padding value
    :param axis: position of the kernel in the array
    """
    assert k & 1, "The size of the kernel should be odd"
    # Get the kernel coordinates
    coord = np.arange(- (k // 2), k // 2 + 1)
    kernel = np.stack(tuple(map(lambda x: x.T, np.meshgrid(coord, coord)[::-1])), axis=-1)
    kernel = kernel.reshape([-1, 2]) * d
    pad_width = kernel[0]
    # Pad the array to ensure the kernel can be applied correctly
    h, w = array.shape[:2]
    array_pad = np.pad(array, constant_values=pad_value,
                       pad_width=np.array(list(-pad_width) + [0] * (array.ndim - 2))[:, None].repeat(2, -1))
    return np.stack([array_pad[y:y + h:s, x:x + w:s] for x, y in kernel - pad_width], axis=axis)


def kernel_func(x1, x2=None, kernel="rbf", args=(1,)):
    # K(x, x') = Φ(x)^T Φ(x')
    x2 = x1 if x2 is None else x2
    # linear function
    if kernel == "linear":
        return x1 @ x2.T
    # polynomial function
    elif kernel == "poly":
        degree, zeta, gamma = args + (1,) * (3 - len(args))
        inpro = x1 @ x2.T
        return np.power(zeta + inpro * gamma, degree)
    # radial basis function
    elif kernel == "rbf":
        sigma = args[0]
        dist = np.square(x1[:, None] - x2).sum(axis=-1)
        return np.exp(- dist / (2 * sigma ** 2))
    raise ValueError(f"Unknown kernel function {kernel}")


# from sklearn.kernel_ridge import KernelRidge
class KernelRegressor:

    def __init__(self, x, y, kernel="rbf", args=(1,)):
        self.y = y
        self.kf = partial(kernel_func, x2=x, kernel=kernel, args=args)

    def predict(self, x):
        k = self.kf(x)
        w = k / k.sum(axis=-1, keepdims=True)
        return (w * self.y).sum(axis=-1)


if __name__ == "__main__":
    np.set_printoptions(3, suppress=True)

    x = (np.random.random([100, 1]) - 0.5) * 10
    y = np.sin(x).flatten()

    plt.scatter(x.flatten(), y)

    kr = KernelRegressor(x, y, args=(0.5,))
    x_ = np.linspace(x.min(), x.max(), 100)
    fx = kr.predict(x_[:, None])

    plt.plot(x_, fx)
    plt.show()
