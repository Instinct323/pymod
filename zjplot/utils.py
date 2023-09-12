import logging

import matplotlib.pyplot as plt

# matplotlib 颜色常量
red = 'orangered'
orange = 'orange'
yellow = 'yellow'
green = 'greenyellow'
cyan = 'aqua'
blue = 'deepskyblue'
purple = 'mediumpurple'
pink = 'violet'
rainbow = [red, orange, yellow, green, blue, purple, pink]

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def figure3d():
    figure = plt.subplot(projection='3d')
    tuple(getattr(figure, f'set_{i}label')(i) for i in 'xyz')
    return figure


def std_coord(*args, zero_p=True):
    ''' 显示标准轴
        args: subplot 参数'''
    fig = plt.subplot(*args)
    for key in 'right', 'top':
        fig.spines[key].set_color('None')
    if zero_p:
        for key in 'left', 'bottom':
            fig.spines[key].set_position(('data', 0))
    return fig


def boxplot(dataset, labels, colors):
    ''' 绘制箱线图'''
    bp = plt.boxplot(dataset, labels=labels)
    for i, color in enumerate(colors):
        bp['boxes'][i].set(color=color, linewidth=1.5)
        bp['medians'][i].set(color=color, linewidth=2.1)


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
    pass
