import numpy as np


def skew_symm3(x):
    """ 反对称矩阵 - x(..., 3)"""
    assert x.shape[-1] == 3
    mat = np.zeros(x.shape + (3,), dtype=x.dtype)
    mat[..., 2, 1], mat[..., 0, 2], mat[..., 1, 0] = np.split(x, 3, axis=-1)
    mat[..., 1, 2], mat[..., 2, 0], mat[..., 0, 1] = np.split(-x, 3, axis=-1)
    return mat


def rotvec2mat(rotvec):
    """ 罗德里格斯公式: 旋转向量 -> 旋转矩阵"""
    assert rotvec.shape == (3,)
    theta = np.linalg.norm(rotvec)
    n = rotvec / theta
    cos = np.cos(theta)
    return cos * np.eye(3) + (1 - cos) * (n[:, None] * n) + np.sin(theta) * skew_symm3(n)


class Quat:
    """ 四元数"""
    real = property(lambda self: self.data[..., :1])
    imag = property(lambda self: self.data[..., 1:])

    def __init__(self, data):
        self.data = data if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
        assert self.data.shape[-1] == 4

    def conj(self):
        return __class__(np.concatenate([self.real, - self.imag], axis=-1))

    def inv(self):
        return self.conj() / np.square(abs(self))

    def __abs__(self):
        return np.linalg.norm(self.data, axis=-1, keepdims=True)

    def __add__(self, other):
        return __class__(self.data + (other.data if isinstance(other, __class__) else other))

    def __sub__(self, other):
        return __class__(self.data - (other.data if isinstance(other, __class__) else other))

    def __neg__(self):
        return __class__(- self.data)

    def __mul__(self, other):
        return __class__(
            np.concatenate([
                self.real * other.real - (self.imag * other.imag).sum(axis=-1, keepdims=True),
                self.real * other.imag + self.imag * other.real + np.cross(self.imag, other.imag)
            ], axis=-1) if isinstance(other, __class__) else self.data * other)

    def __truediv__(self, other):
        return __class__(
            (self * other.inv()).data if isinstance(other, __class__) else self.data / other)

    def __repr__(self):
        return str(self.data)

    def to_rotvec(self):
        ret = self.imag / np.linalg.norm(self.imag, axis=-1, keepdims=True)
        return ret * 2 * np.arccos(self.real)

    @classmethod
    def from_rotvec(cls, rotvec):
        assert rotvec.shape[-1] == 3
        theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
        return np.concatenate([np.cos(theta / 2), rotvec / theta], axis=-1)


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    q = Quat([1, 2, 3, 4])
    q /= abs(q)
    rotvec = q.to_rotvec()
    print(rotvec)
    print(rotvec2mat(rotvec))
    print(Rotation.from_rotvec(rotvec).as_matrix())
