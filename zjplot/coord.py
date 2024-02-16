import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation


class _SEnd:
    dtype = np.float32
    dim = None
    # 位置, 各个轴的方向向量
    position = property(lambda self: self.s[:self.dim, -1])
    direction = property(lambda self: self.s[:self.dim, :self.dim])

    def __init__(self, csys: np.ndarray = None):
        size = self.dim + 1
        self.s = np.eye(size, dtype=self.dtype)
        # 使用非空形参
        if isinstance(csys, np.ndarray):
            assert csys.shape == self.s.shape
            self.s = csys

    def abs_tf(self, tf):
        ''' 绝对变换'''
        return type(self)(tf @ self.s)

    def rela_tf(self, tf):
        ''' 相对变换'''
        return type(self)(self.s @ tf)

    def apply(self, *coords) -> tuple:
        ''' 局部坐标值 -> 全局坐标值'''
        xyz = np.stack(coords, axis=-1) @ self.direction.T + self.position
        return tuple(i[..., 0] for i in np.split(xyz, self.dim, axis=-1))

    def plot_coord_sys(self, length=.5, linewidth=None,
                       colors=['orangered', 'deepskyblue', 'greenyellow'], labels='xyz'):
        ''' 绘制局部坐标系'''
        pos = self.position
        axis = self.direction.T * length
        for i in range(self.dim):
            plt.plot(*zip(pos, pos + axis[i]), c=colors[i], label=labels[i], linewidth=linewidth)

    def __repr__(self):
        return str(self.s) + '\n'


class SE2d(_SEnd):
    dim = 2

    def apply(self, x: np.ndarray, y: np.ndarray) -> tuple:
        ''' 局部坐标值 -> 全局坐标值'''
        return super().apply(x, y)

    def transform(self, dx: float = 0., dy: float = 0.,
                  theta: float = 0, relative: bool = True):
        ''' :param dx,dy: 平移变换的参数
            :param theta: 旋转变换的参数
            :param relative: 是否使用相对变换'''
        mat = np.concatenate((np.eye(3, 2, dtype=self.dtype),
                              np.array((dx, dy, 1))[:, None]), axis=-1)
        if theta:
            theta = np.deg2rad(theta)
            cos, sin = np.cos(theta), np.sin(theta)
            mat[:2, :2] = np.array([[cos, -sin], [sin, cos]])
        return (self.rela_tf if relative else self.abs_tf)(mat)


class SE3d(_SEnd):
    dim = 3

    def apply(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
        ''' 局部坐标值 -> 全局坐标值'''
        return super().apply(x, y, z)

    @classmethod
    def trans(cls, dx: float = 0., dy: float = 0., dz: float = 0.) -> np.ndarray:
        ''' 齐次变换矩阵 - 平移'''
        return np.concatenate((np.eye(4, 3, dtype=cls.dtype),
                               np.array((dx, dy, dz, 1))[:, None]), axis=-1)

    @classmethod
    def rot(cls, yaw=0, pitch=0, roll=0) -> np.ndarray:
        ''' 齐次变换矩阵 - 旋转
            :param yaw: 偏航角, 绕 z 轴旋转
            :param pitch: 俯仰角, 绕 y 轴旋转
            :param roll: 滚转角, 绕 x 轴旋转

            :example:
            >>> rpy = [30, 20, 10]
            >>> rot = SE3d.rot

            >>> a = rot(*rpy[::-1])
            >>> b = rot(yaw=rpy[2]) @ rot(pitch=rpy[1]) @ rot(roll=rpy[0])
            >>> np.square(a - b).sum()
            8.049117e-15'''
        mat = np.eye(4, dtype=cls.dtype)
        mat[:3, :3] = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
        return mat


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rot = SE3d.rot
    trans = SE3d.trans

    se = SE3d().abs_tf(rot(30, 20)).abs_tf(trans(1, 2, 3))

    fig = plt.subplot(projection='3d')

    p = np.array([1, 2, 3])
    for i in range(10):
        se.s[:3, :3] *= 1.1
        fig.scatter(*se.apply(*p), c='r')

    plt.show()
