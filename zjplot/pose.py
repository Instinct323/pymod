import numpy as np

from scipy.spatial import transform

SO3 = transform.Rotation


class _SEn:
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

    def __matmul__(self, other):
        return self.rela_tf(other if isinstance(other, np.ndarray) else other.s)

    def abs_tf(self, tf):
        """ 绝对变换"""
        return type(self)(tf @ self.s)

    def rela_tf(self, tf):
        """ 相对变换"""
        return type(self)(self.s @ tf)

    def apply(self, *coords) -> tuple:
        """ 局部坐标值 -> 全局坐标值"""
        xyz = np.stack(coords, axis=-1) @ self.direction.T + self.position
        return tuple(i[..., 0] for i in np.split(xyz, self.dim, axis=-1))

    def plot_coord_sys(self, length=.5, linewidth=None,
                       colors=["orangered", "deepskyblue", "greenyellow"], labels="xyz"):
        """ 绘制局部坐标系"""
        pos = self.position
        axis = self.direction.T * length
        for i in range(self.dim):
            plt.plot(*zip(pos, pos + axis[i]), c=colors[i], label=labels[i], linewidth=linewidth)
        plt.axis("equal")

    def __repr__(self):
        return str(self.s) + "\n"


class SE2(_SEn):
    dim = 2

    def apply(self, x: np.ndarray, y: np.ndarray) -> tuple:
        return super().apply(x, y)

    def transform(self, dx: float = 0., dy: float = 0.,
                  theta: float = 0, relative: bool = True):
        """ :param dx,dy: 平移变换的参数
            :param theta: 旋转变换的参数
            :param relative: 是否使用相对变换"""
        mat = np.concatenate((np.eye(3, 2, dtype=self.dtype),
                              np.array((dx, dy, 1))[:, None]), axis=-1)
        if theta:
            theta = np.deg2rad(theta)
            cos, sin = np.cos(theta), np.sin(theta)
            mat[:2, :2] = np.array([[cos, -sin], [sin, cos]])
        return (self.rela_tf if relative else self.abs_tf)(mat)


class SE3(_SEn):
    dim = 3

    def as_vec7(self):
        return np.concatenate((self.position, SO3.from_matrix(self.direction).as_quat()))

    def apply(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
        return super().apply(x, y, z)

    def get_3DGS(self, size, scale=(1, 1, 1e-4)) -> np.ndarray:
        L = self.direction @ np.diag(scale)
        return np.random.multivariate_normal(self.position, L @ L.T, size)

    @classmethod
    def trans(cls, dx: float = 0., dy: float = 0., dz: float = 0.) -> np.ndarray:
        return np.concatenate((np.eye(4, 3, dtype=cls.dtype),
                               np.array((dx, dy, dz, 1))[:, None]), axis=-1)

    @classmethod
    def rot(cls, yaw=0, pitch=0, roll=0) -> np.ndarray:
        """ :param yaw: 偏航角, 绕 z 轴旋转
            :param pitch: 俯仰角, 绕 y 轴旋转
            :param roll: 滚转角, 绕 x 轴旋转

            :example:
            >>> rpy = [30, 20, 10]
            >>> rot = SE3.rot

            >>> a = rot(*rpy[::-1])
            >>> b = rot(yaw=rpy[2]) @ rot(pitch=rpy[1]) @ rot(roll=rpy[0])
            >>> np.square(a - b).sum()
            8.049117e-15"""
        mat = np.eye(4, dtype=cls.dtype)
        mat[:3, :3] = SO3.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()
        return mat


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rot = SE3.rot
    trans = SE3.trans

    se = SE3().rela_tf(rot(30, 20)).abs_tf(trans(1, 2, 3))

    fig = plt.subplot(projection="3d")

    fig.scatter(*se.get_3DGS(3000, scale=(1, 2, 0)).T, color="gray", s=10)
    se.plot_coord_sys(3)

    plt.show()
