import functools

import matplotlib.pyplot as plt
import numpy as np


def kernel_select(img, k, s=1, d=1, pad_value=0, axis=0):
    ''' img: OpenCV 格式的图像 [h, w, c]
        k: kernel size
        s: stride
        d: dilation
        pad_value: 边界填充常量
        axis: 新维度的位置'''
    assert k & 1, 'The size of the kernel should be odd'
    # 获取膨胀操作核
    coord = np.arange(- (k // 2), k // 2 + 1)
    kernel = np.stack(tuple(map(lambda x: x.T, np.meshgrid(coord, coord)[::-1])), axis=-1)
    kernel = kernel.reshape([-1, 2]) * d
    pad_width = kernel[0]
    # 填充图像的边界
    h, w = img.shape[:2]
    img_pad = np.pad(img, constant_values=pad_value,
                     pad_width=np.append(-pad_width, 0)[:, None].repeat(2, -1))
    return np.stack([img_pad[y:y + h:s, x:x + w:s] for x, y in kernel - pad_width], axis=axis)


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
