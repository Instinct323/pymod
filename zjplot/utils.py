import logging

import matplotlib.pyplot as plt

BGR_COLOR = {
    'red': (58, 0, 255),
    'orange': (49, 125, 237),
    'yellow': (255, 255, 0),
    'green': (71, 173, 112),
    'cyan': (255, 255, 0),
    'blue': (240, 176, 0),
    'purple': (209, 0, 152),
    'pink': (241, 130, 234),
    'gray': (190, 190, 190),
    'black': (0, 0, 0)
}

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


def hotmap(array, cmap='Blues', annotate=10, colorbar=False,
           xticks=None, yticks=None, xlabel=None, ylabel=None):
    # 去除坐标轴
    fig = plt.subplot()
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
            plt.annotate(f'{item:.2f}', (j - .1, i + .05), size=annotate)
    # 坐标轴标签
    plt.xticks(range(len(array)), xticks)
    plt.yticks(range(len(array[0])), yticks, rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == '__main__':
    pass
