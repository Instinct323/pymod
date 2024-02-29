import time

import matplotlib.patches as pch

# coord.py 详见: https://blog.csdn.net/qq_55745968/article/details/129912954
from .se import SE3d, SE2d
from .utils import *

ROUND_EDGE = 30  # 圆等效多边形边数
DTYPE = np.float16  # 矩阵使用的数据类型


class Spot:
    """ 闪烁斑点对象
        :param xylim: xy 坐标区间, [[xmin, ymin], [xmax, ymax]]
        :param n: 闪烁斑点的数量
        :param r: 斑点的半径均值, 标准差为 r/2, 最小值为 r/4
        :param delta: 斑点寿命的均值, 标准差为 delta, 最小值为 delta/10"""
    colors = [red, orange, yellow, green, cyan, blue, purple, pink]

    def __init__(self, xylim: np.ndarray, n: int,
                 r: float = .2, delta: float = 1., alpha: float = .7):
        # <群体属性>
        self.xylim = xylim
        self.n, self.r = n, r
        self.delta = delta
        self.alpha = alpha
        # <个体属性>
        # 出生时间, 寿命
        self.start = np.array([])
        self.deltas = np.array([])
        # 出生位置, 半径, 颜色
        self.xy = np.ones([0, 2])
        self.radius = np.array([])
        self.color = np.array([])
        # 生产斑点
        self.produce()

    def scatter(self, n):
        return self.xylim[0] + np.random.rand(n, 2) * (self.xylim[1] - self.xylim[0])

    def produce(self, filt=None):
        # 筛除生存时间耗尽的斑点
        if isinstance(filt, np.ndarray):
            for key in ("start", "color", "xy", "radius", "deltas"):
                setattr(self, key, getattr(self, key)[filt])
        # 补全缺失的斑点
        lack = self.n - self.xy.shape[0]
        if lack > 0:
            # 记录出生时间, 寿命
            self.start = np.concatenate((self.start, np.full(lack, fill_value=time.time())))
            self.deltas = np.concatenate((self.deltas, np.maximum(np.random.normal(
                loc=self.delta, scale=self.delta, size=lack), self.delta / 10)))
            # 随机位置, 随机半径, 随机颜色
            self.xy = np.concatenate((self.xy, self.scatter(lack)), axis=0)
            self.radius = np.concatenate((self.radius, np.maximum(np.random.normal(
                loc=self.r, scale=self.r / 2, size=lack), self.r / 4)))
            self.color = np.concatenate((self.color, np.random.choice(self.colors, size=lack)))

    def __call__(self, fig, csys: SE2d = None):
        """ 刷新斑点的透明度
            :param csys: SE2d 对象"""
        x = time.time() - self.start
        # y = 4/d^2 x (d - x)
        alpha = self.alpha * np.maximum(4 / self.deltas ** 2 * x * (self.deltas - x), 0)
        # 向图像添加斑点
        for i, xy in enumerate(np.stack(csys.apply(*self.xy.T), axis=-1) if csys else self.xy):
            patch = pch.Circle(xy, self.radius[i], alpha=alpha[i], edgecolor=None, facecolor=self.color[i])
            fig.add_patch(patch)
        self.produce(alpha > 0)


def cylinder(figure, csys: SE3d,
             R: float, h: float, r: float = 0,
             smooth: int = 2, **plot_kwd):
    """ 以 csys 的 z 轴为主轴绘制圆柱
        :param figure: 3D 工作站对象
        :param csys: SE3d 齐次坐标系
        :param R: 圆柱底面外径
        :param r: 圆柱底面内径
        :param h: 圆柱高度
        :param smooth: 图像细致程度 (至少 2)"""
    theta = np.linspace(0, 2 * np.pi, ROUND_EDGE, dtype=DTYPE)
    z = np.linspace(-h / 2, h / 2, smooth, dtype=DTYPE)
    theta, z = np.meshgrid(theta, z)
    # 绘制圆柱内外曲面: 以 z 轴为主轴, 原点为中心
    x, y = np.cos(theta), np.sin(theta)
    figure.plot_surface(*csys.apply(x * R, y * R, z), **plot_kwd)
    figure.plot_surface(*csys.apply(x * r, y * r, z), **plot_kwd)

    phi = np.linspace(0, 2 * np.pi, ROUND_EDGE, dtype=DTYPE)
    radius = np.linspace(r, R, 2, dtype=DTYPE)
    phi, radius = np.meshgrid(phi, radius)
    # 绘制上下两底面: 法向量为 z 轴, 原点为中心, 在 z 轴上偏移得到两底面
    x, y = np.cos(phi) * radius, np.sin(phi) * radius
    z = np.zeros_like(x)
    for dz in (-h / 2, h / 2):
        s = csys.rela_tf(SE3d.trans(dz=dz))
        figure.plot_surface(*s.apply(x, y, z), **plot_kwd)


def ball(figure, csys: SE3d, r: float, **plot_kwd):
    """ 绘制球体
        :param figure: 3D 工作站对象
        :param csys: SE3d 齐次坐标系
        :param r: 球体半径"""
    theta = np.linspace(0, 2 * np.pi, ROUND_EDGE, dtype=DTYPE)
    phi = np.linspace(0, np.pi, ROUND_EDGE // 2, dtype=DTYPE)
    theta, phi = np.meshgrid(theta, phi)
    sin_phi = np.sin(phi) * r
    figure.plot_surface(*csys.apply(np.cos(theta) * sin_phi,
                                    np.sin(theta) * sin_phi,
                                    np.cos(phi) * r), **plot_kwd)


def rubik_cube(figure, csys: SE3d,
               length: float, hollow: float = 0.7, smooth: int = 10,
               colors: list = [red, orange, yellow, green, cyan, blue, purple, pink], **plot_kwd):
    """ 绘制魔方
        :param figure: 3D 工作站对象
        :param csys: SE3d 齐次坐标系
        :param length: 边长
        :param smooth: 魔方的细粒度"""
    x = np.linspace(-length / 2, length / 2, smooth + 1)
    filled = np.random.random([smooth] * 3) > hollow
    color = np.random.choice(colors, size=filled.shape)
    # 绘制各个通道
    figure.voxels(*csys.apply(*np.meshgrid(x, x, x)), filled=filled,
                  facecolors=color, edgecolors="white", **plot_kwd)
    return figure


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = [6.4, 6.4]
    index = 1

    if index == 1:
        fig = plt.subplot()
        limit = 8
        plt.xlim([-limit, limit])
        plt.ylim([-limit, limit])
        x = np.array([-2, -1, 0, 1, 2])
        y = np.array([-1, 1, 0, 1, -1])


        def pathpatch(fig, csys=None):
            xy = csys.apply(x, y) if csys else (x, y)
            fig.add_patch(pch.PathPatch(pch.Path(np.stack(xy, axis=-1)),
                                        facecolor="red", alpha=0.5, edgecolor="w"))


        csys = SE2d().transform(dx=2)
        print(csys)
        # pathpatch(fig), plt.show()

        while True:
            plt.cla()
            plt.xlim([-limit, limit])
            plt.ylim([-limit, limit])

            csys = csys.transform(theta=3, relative=False)
            pathpatch(fig, csys)
            # 绘制局部坐标系
            csys.plot_coord_sys(linewidth=3, length=2), plt.legend()

            # 旋转 37°
            csys_new = csys.transform(theta=37).transform(dx=5)
            pathpatch(fig, csys_new)

            plt.pause(0.01)

    elif index == 2:
        fig = plt.subplot()
        # 初始化齐次坐标系
        csys = SE2d()
        # 初始化闪烁斑点
        spot = Spot(xylim=np.array([[-2, 2], [-.5, .5]]).T, n=50, r=.2, delta=1)

        while True:
            fig.cla()
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            # 使当前的齐次坐标系旋转 2°
            csys = csys.transform(theta=2)
            # 调用闪烁斑点的 __call__ 函数绘制
            spot(fig, csys=csys)
            plt.pause(0.01)

    elif index == 3:
        fig = figure3d()
        fig.set_xlim((-6, 4))
        fig.set_ylim((-3, 7))
        fig.set_zlim((-5, 5))

        rot = SE3d.rot
        trans = SE3d.trans

        # 绕 y 轴相对旋转 20°, 再绝对平移
        csys = SE3d().rela_tf(rot(pitch=20)).abs_tf(trans(dx=-1, dy=2, dz=-2))
        print(csys)
        # 以 z 轴为主轴, 绘制空心圆柱
        cylinder(fig, csys=csys, R=5, r=4, h=3, cmap="Set3", alpha=0.5)
        # 绘制局部坐标系
        csys.plot_coord_sys(length=10, linewidth=5), plt.legend()

        # 在空心圆柱的 z 轴上平移
        csys = csys.rela_tf(trans(dz=5))
        print(csys)
        # 绘制空心魔方
        rubik_cube(fig, csys=csys, length=6, hollow=0.8, smooth=10, alpha=0.6)
        plt.show()

    else:
        fig = figure3d()
        fig.set_xticks([], [])
        fig.set_yticks([], [])
        fig.set_zticks([], [])

        rot = SE3d.rot
        trans = SE3d.trans

        csys = SE3d()
        csys.plot_coord_sys(linewidth=5, length=1)

        csys = csys.abs_tf(trans(dx=1, dy=1, dz=1)).rela_tf(rot(pitch=45))
        csys.plot_coord_sys(linewidth=5, labels="noa", colors=[orange, yellow, pink])

        plt.legend()
        plt.show()
