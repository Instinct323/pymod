import numpy as np


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
