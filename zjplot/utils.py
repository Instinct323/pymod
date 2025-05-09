import logging
from pathlib import Path
from typing import Union, Tuple

import matplotlib
import matplotlib.patches as pch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# matplotlib 颜色常量
red = "orangered"
orange = "orange"
yellow = "gold"
green = "greenyellow"
cyan = "aqua"
blue = "deepskyblue"
purple = "mediumpurple"
pink = "violet"
rainbow = [red, orange, yellow, green, cyan, blue, purple, pink]

fav1 = ["#0ddbf5", "#1d9bf7", "#8386fc", "#303cf9", "#fe5357", "#fd7c1a", "#ffbd15", "#fcff07"]
fav2 = ["#444577", "#c65861", "#f3dee0", "#ffa725", "#ff6b62", "#be588d", "#58538b"]
fav3 = ["#cb78a6", "#d35f00", "#f7ec44", "#009d73", "#fcb93e", "#0072b2", "#979797"]

# fontdict: 小五号宋体
SIMSUN = {"fontsize": 9, "family": "SimSun"}
plt.rcParams["font.sans-serif"] = "Microsoft YaHei"


# plt.rcParams.update({
#     "font.family": "Times New Roman",  # 绘制文本的字体系列
#     "font.size": 9,  # 文本大小 (小五号)
#     "mathtext.fontset": "stix"  # 数学文本渲染
# })


def rand_colors(n=1, cmap=rainbow, seed=0):
    np.random.seed(seed)
    if cmap:
        ret = cmap[:min(n, len(cmap))]
        if len(ret) < n:
            ret += rand_colors(n - len(ret), cmap=None)
        return ret
    return np.random.random([n, 3]).tolist()


def colormap(scale, cmap="Spectral_r", normalize=True):
    cmap = matplotlib.colormaps.get_cmap(cmap)
    if normalize: scale = (scale - scale.min()) / (scale.max() - scale.min())
    return (cmap(scale)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def figure3d():
    figure = plt.subplot(projection="3d")
    tuple(getattr(figure, f"set_{i}label")(i) for i in "xyz")
    return figure


def fig2img(fig: plt.Figure = None):
    """ 将 plt.Figure 对象转换为图像矩阵"""
    if fig is None: fig = plt.gcf()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def crop_white(file: Path):
    import cv2
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(255 - gray)
    cv2.imwrite(file, img[y:y + h, x:x + w])


def pie_kwd(labels, decimal=2, colors=None):
    """ 饼图的关键字参数
        :param labels: 标签
        :param decimal: 百分数精度"""
    return dict(labels=labels,
                colors=colors if colors else rand_colors(len(labels)),
                autopct=lambda x: f"{x:.{decimal}f}%",
                shadow=False,
                explode=(0.05,) * len(labels))


def std_coord(*args, zero_p=True):
    """ 显示标准轴
        :param args: subplot 参数"""
    fig = plt.subplot(*args)
    for key in "right", "top":
        fig.spines[key].set_color("None")
    if zero_p:
        for key in "left", "bottom":
            fig.spines[key].set_position(("data", 0))
    return fig


def boxplot(dataset, labels=None, colors=None):
    """ 绘制箱线图"""
    bp = plt.boxplot(dataset, labels=labels, patch_artist=True)
    for i, color in enumerate(
            colors if colors else rand_colors(len(bp["boxes"]))):
        bp["boxes"][i].set(color=color, linewidth=1.5)
        bp["medians"][i].set(color="white", linewidth=2.1)
    return bp


def violinplot(dataset: list, labels=None, colors=None,
               alpha=.4, linewidth=3, xrotate=0, yrotate=0):
    """ 绘制小提琴图"""
    for data in dataset: data.sort()
    vp = plt.violinplot(dataset, showextrema=False, widths=0.8)
    colors = colors if colors else rand_colors(len(dataset))
    for i, bd in enumerate(vp["bodies"]):
        bd.set(color=colors[i], alpha=alpha, linewidth=0)
    # 添加标签
    x = np.arange(1, 1 + len(dataset))
    if labels: plt.xticks(x, labels, rotation=xrotate)
    plt.yticks(rotation=yrotate)
    # 在中位线处绘制散点, 25-75 间绘制粗线, 0-100 间绘制细线
    q = np.array([np.percentile(data, [0, 25, 50, 75, 100]) for data in dataset]).T
    plt.vlines(x, q[0], q[-1], colors=colors, lw=linewidth)
    plt.vlines(x, q[1], q[-2], colors=colors, lw=linewidth * 3)
    plt.scatter(x, q[2], color="white", s=linewidth * 18, zorder=3)
    return vp


def regionplot(y, mean, std, y_color=blue,
               region_color=None, region_alpha=.2, label=None, sample=100):
    """ 绘制区域图"""
    sample = min(sample, len(y))
    x = np.linspace(0, len(y) - 1, sample, dtype=np.int32)
    y, mean, std = y[x], mean[x], std[x]

    plt.plot(x, y, color=y_color, label=label)
    plt.plot(x, mean, color="white")
    plt.fill_between(x, mean - std, mean + std,
                     color=region_color if region_color else y_color, alpha=region_alpha)


def bar2d(dataset, xticks=None, labels=None, colors=None, alpha=1):
    x = np.arange(dataset.shape[1])
    bias = np.linspace(-.5, .5, dataset.shape[0] + 2)[1:-1]
    w = .7 * (.5 - bias[-1])
    # 处理缺失值信息
    labels = labels or (None,) * len(bias)
    colors = colors or (None,) * len(bias)
    for i, y in enumerate(dataset):
        plt.bar(x + bias[i], y, width=w, label=labels[i], color=colors[i], alpha=alpha)
    # 绘制标签信息
    if any(labels): plt.legend()
    if xticks: plt.xticks(x, xticks)


def hotmap(array, fig=None, pos=0, fformat="%f", cmap="Blues", size=10, title=None, colorbar=False,
           xticks=None, yticks=None, xlabel=None, ylabel=None, xrotate=0, yrotate=90):
    pos = np.array([-.1, .05]) + pos
    # 去除坐标轴
    fig = fig or plt.subplot()
    plt.title(title)
    for key in "right", "top", "left", "bottom":
        fig.spines[key].set_color("None")
    fig.xaxis.set_ticks_position("top")
    fig.xaxis.set_label_position("top")
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


def corrplot(df: pd.DataFrame,
             size: np.ndarray = None,
             vmin: float = -1,
             vmax: float = 1,
             cmap: str = "coolwarm"):
    """ 绘制相关系数矩阵图
        :param df: 相关系数矩阵
        :param size: 每个单元格的直径 (上三角)"""
    assert df.index.equals(df.columns), "Index and columns must be the same"
    # 显示上三角部分的相关系数标注
    ax = sns.heatmap(df * 0, vmax=1, annot=df, fmt=".2f", cbar=False, cmap="binary", square=True,
                     mask=np.tril(np.ones_like(df, dtype=bool)))
    r = (np.ones_like(df) if size is None else size) / 2
    # 获取颜色映射对象
    cmap = plt.get_cmap(cmap)
    # 创建一个归一化对象，用于将相关系数映射到颜色
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # 创建一个标量映射对象，用于创建颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    # 绘制相关系数矩阵的下三角部分
    df = df.to_numpy()
    for i in range(df.shape[0]):
        for j in range(i, df.shape[1]):
            # 在每个单元格中绘制一个圆形，颜色根据相关系数值确定
            ax.add_patch(pch.Circle((i + .5, j + .5), r[i, j], facecolor=cmap(norm(df[i, j]))))
    return ax


def threshold_scatter(x: np.ndarray,
                      y: np.ndarray,
                      z: np.ndarray,
                      threshold: Union[float, Tuple[float, float]],
                      cmap: str = "coolwarm",
                      size: Tuple[float, float] = (5, 40),
                      eps: float = 1e-8):
    """ 绘制阈值散点图"""
    if hasattr(threshold, "__len__"):
        threshold = np.array(sorted(threshold))
        if len(threshold) == 2:
            threshold = np.array([[1, 0], [.5, .5], [0, 1]]) @ threshold
        # 把阈值分为 3 个区间
        assert len(threshold) == 3, f"Invalid threshold: {threshold}"
        radio = np.abs(z - threshold[0]) / (threshold[-1] - threshold[0])
    else:
        radio = z / threshold
    norm = plt.Normalize(vmin=0, vmax=2)
    radio = norm(np.clip(radio, eps, 2 - eps))
    s = size[0] + (size[1] - size[0]) * radio
    ax = sns.scatterplot(x=x, y=y, s=s, c=plt.get_cmap(cmap)(radio))
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    return ax


def grading_scatter(samples: pd.DataFrame,
                    bounds: list,
                    cmap: str = "autumn_r",
                    size: Tuple[float, float] = (5, 40),
                    float_fmt="%s"):
    """ 分级散点图
        :param samples: 样本数据 (x, y, z)
        :param cmap: 颜色映射
        :param bounds: 分级边界
        :param size: 散点大小范围
        :param float_fmt: 浮点数格式"""
    col = samples.columns[-1]
    # 获取颜色映射和大小
    n = len(bounds) + 1
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    size = np.logspace(*np.log2(size), n, base=2)
    # 根据 bounds 的有效值部分选择颜色、大小
    mask = np.bool_(bounds)
    bounds = np.array(bounds)[mask]
    mask = np.concatenate([[True], mask])
    colors = colors[mask]
    size = size[mask]
    # 获取分级掩码矩阵
    x = samples[col]
    labels = ([f"<{float_fmt}" % bounds[0]] +
              [(f"{float_fmt}~{float_fmt}" % (bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)] +
              [f">{float_fmt}" % bounds[-1]])
    bounds = [-np.inf] + list(bounds) + [np.inf]
    mask = np.stack([(bounds[i] <= x) & (x < bounds[i + 1]) for i in range(len(bounds) - 1)], axis=1)
    mask = pd.DataFrame(mask, columns=labels, index=x.index)
    # 绘制分级散点图
    for j, label in enumerate(mask.columns):
        tar = samples[mask[label]]
        sns.scatterplot(tar, x="x", y="y", label=label, color=colors[j], s=size[j])
    plt.legend(title=col)


def residplot(x: np.ndarray,
              y: np.ndarray,
              pred: np.ndarray,
              pred_color="deepskyblue",
              line_color="gray",
              gt_cmap="spring",
              size=(5, 40)):
    # Pred
    plt.plot(x, pred, color=pred_color, zorder=0)
    # Residual
    cmap = plt.get_cmap(gt_cmap)
    res = np.abs(y - pred)
    s = res / res.max()
    plt.vlines(x, y, pred, colors=line_color, linestyles='--', linewidth=1, zorder=-1)
    return sns.scatterplot(x=x, y=y, color=cmap(s), s=size[0] + (size[1] - size[0]) * s, zorder=1)


class PltVideo:
    """ :param fig_id: plt.figure 的 id
        :param video_writer: zjcv.VideoWriter 对象

        :example:
        >>> import cv2
        >>> from pymod.utils import zjcv
        >>>
        >>> def draw(pv):
        ...     plt.figure(100)
        ...     plt.clf()
        ...     plt.scatter(np.arange(100), np.random.random(100))
        ...     pv.write()
        ...     plt.pause(1e-3)
        >>>
        >>> with PltVideo(100, zjcv.VideoWriter(r"C:\Downloads\demo.mp4", cvt_color=cv2.COLOR_RGB2BGR)) as pv:
        ...     for i in range(90): draw(pv)
        """

    def __init__(self,
                 fig_id: int,
                 video_writer: "VideoWriter"):
        self.fig_id = fig_id
        for k in "write", "save", "__enter__", "__exit__":
            assert hasattr(video_writer, k), f"{type(video_writer).__name__} has no function {k}"
        self.video_writer = video_writer

    def write(self):
        self.video_writer.write(fig2img(plt.figure(self.fig_id)))

    def save(self):
        self.video_writer.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
        if exc_type: return False
        return self


if __name__ == "__main__":
    threshold_scatter(np.random.random(100), np.random.random(100), np.random.random(100) * 2, (0.3, 1, 1.7))
    plt.show()
