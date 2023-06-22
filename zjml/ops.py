import numpy as np
import torch

BGR_COLOR = {'red': (58, 0, 255),
             'orange': (49, 125, 237),
             'yellow': (255, 255, 0),
             'green': (71, 173, 112),
             'cyan': (255, 255, 0),
             'blue': (240, 176, 0),
             'purple': (209, 0, 152),
             'pink': (241, 130, 234),
             'gray': (190, 190, 190),
             'black': (0, 0, 0)}


def make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.linspace(0.5 / ny, 1 - 0.5 / ny, ny),
                             torch.linspace(0.5 / nx, 1 - 0.5 / nx, nx)])
    return torch.stack([xv, yv], -1)


def plot_conv(fig, width, height, colors, alpha=0.7,
              xbias=0, ybias=0, zbias=0):
    ''' colors: 各个通道的颜色'''
    x = np.arange(width + 1) + xbias - 0.5
    y = np.arange(height + 1) + ybias - 0.5
    filled = np.ones([width, height, 1])
    # 绘制各个通道
    for i, color in enumerate(colors):
        z = np.array([i + 1, i]) + zbias - 0.5
        pos = np.meshgrid(x, y, z)
        fig.voxels(*pos, alpha=alpha, filled=filled,
                   facecolors=color, edgecolors='white')
    return fig


def plot_fc(fig, in_features, out_features, g=1, color='silver',
            size=1000, marker='.', alpha=1, **kwargs):
    ''' 绘制全连接层散点、权值'''
    dims = len(in_features[0])
    if g == 1:
        all_f = np.concatenate([in_features, out_features], axis=0).T
        # 绘制神经元散点
        fig.scatter(*[all_f[i] for i in range(dims)],
                    color=color, s=size, marker=marker, alpha=alpha, **kwargs)
        # 绘制神经元连接
        for in_f in in_features:
            for out_f in out_features:
                connect = np.stack([in_f, out_f]).T
                fig.plot(*[connect[i] for i in range(dims)],
                         color=color, alpha=alpha, **kwargs)
    else:
        # 对神经元进行分组
        in_features = in_features.reshape(g, -1, dims)
        out_features = out_features.reshape(g, -1, dims)
        # 分组绘制全连接层
        for in_f, out_f in zip(in_features, out_features):
            plot_fc(fig, in_f, out_f, color=color,
                    size=size, marker=marker, alpha=alpha, **kwargs)
    return fig


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


if __name__ == '__main__':
    print(make_grid())
