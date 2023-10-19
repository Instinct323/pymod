import logging

import matplotlib.pyplot as plt
import numpy as np

# matplotlib 颜色常量
red = 'orangered'
orange = 'orange'
yellow = 'gold'
green = 'greenyellow'
cyan = 'aqua'
blue = 'deepskyblue'
purple = 'mediumpurple'
pink = 'violet'
rainbow = [red, orange, yellow, green, cyan, blue, purple, pink]

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def rand_colors(n=1, cmap=rainbow, seed=0):
    np.random.seed(seed)
    if cmap:
        ret = cmap[:min(n, len(cmap))]
        if len(ret) < n:
            ret += rand_colors(n - len(ret), cmap=None)
        return ret
    return np.random.random([n, 3]).tolist()


def figure3d():
    figure = plt.subplot(projection='3d')
    tuple(getattr(figure, f'set_{i}label')(i) for i in 'xyz')
    return figure


def std_coord(*args, zero_p=True):
    ''' 显示标准轴
        :param args: subplot 参数'''
    fig = plt.subplot(*args)
    for key in 'right', 'top':
        fig.spines[key].set_color('None')
    if zero_p:
        for key in 'left', 'bottom':
            fig.spines[key].set_position(('data', 0))
    return fig


def boxplot(dataset, labels=None, colors=None):
    ''' 绘制箱线图'''
    bp = plt.boxplot(dataset, labels=labels)
    for i, color in enumerate(
            colors if colors else rand_colors(len(bp['boxes']))):
        bp['boxes'][i].set(color=color, linewidth=1.5)
        bp['medians'][i].set(color=color, linewidth=2.1)
    return bp


def violinplot(dataset: list, labels=None, colors=None,
               alpha=.4, linewidth=3, xrotate=0, yrotate=0):
    ''' 绘制小提琴图'''
    for data in dataset: data.sort()
    vp = plt.violinplot(dataset, showextrema=False, widths=0.8)
    colors = colors if colors else rand_colors(len(dataset))
    for i, bd in enumerate(vp['bodies']):
        bd.set(color=colors[i], alpha=alpha, linewidth=0)
    # 添加标签
    x = np.arange(1, 1 + len(dataset))
    if labels: plt.xticks(x, labels, rotation=xrotate)
    plt.yticks(rotation=yrotate)
    # 在中位线处绘制散点, 25-75 间绘制粗线, 0-100 间绘制细线
    q = np.array([np.percentile(data, [0, 25, 50, 75, 100]) for data in dataset]).T
    plt.vlines(x, q[0], q[-1], colors=colors, lw=linewidth)
    plt.vlines(x, q[1], q[-2], colors=colors, lw=linewidth * 3)
    plt.scatter(x, q[2], color='white', s=linewidth * 18, zorder=3)
    return vp


def bar2d(dataset, xticks=None, labels=None, colors=None, alpha=1):
    x = np.arange(dataset.shape[1])
    bias = np.linspace(-.5, .5, dataset.shape[0] + 2)[1:-1]
    w = .7 * (.5 - bias[-1])
    # 处理缺失值信息
    labels = [None] * len(bias) if labels is None else labels
    colors = [None] * len(bias) if colors is None else colors
    for i, y in enumerate(dataset):
        plt.bar(x + bias[i], y, width=w, label=labels[i], color=colors[i], alpha=alpha)
    # 绘制标签信息
    if any(labels): plt.legend()
    if xticks: plt.xticks(x, xticks)


def hotmap(array, fig=None, pos=0, fformat='%f', cmap='Blues', size=10, title=None, colorbar=False,
           xticks=None, yticks=None, xlabel=None, ylabel=None, xrotate=0, yrotate=90):
    pos = np.array([-.1, .05]) + pos
    # 去除坐标轴
    fig = plt.subplot() if fig is None else fig
    plt.title(title)
    for key in 'right', 'top', 'left', 'bottom':
        fig.spines[key].set_color('None')
    fig.xaxis.set_ticks_position('top')
    fig.xaxis.set_label_position('top')
    # 显示热力图
    plt.imshow(array, cmap=cmap)
    if colorbar: plt.colorbar()
    # 标注数据信息
    for i, row in enumerate(array):
        for j, item in enumerate(row):
            if np.isfinite(item):
                plt.annotate(fformat % item, pos + [j, i], size=size)
    # 坐标轴标签
    plt.xticks(range(len(array[0])), xticks, rotation=xrotate)
    plt.yticks(range(len(array)), yticks, rotation=yrotate)
    plt.xlabel(xlabel), plt.ylabel(ylabel)


if __name__ == '__main__':
    print(rand_colors(10))
