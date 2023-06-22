from .plot import *


def trans(dx, dy, dz):
    ''' 齐次变换矩阵: 平移'''
    mat = np.eye(4, dtype=ARRAY_TYPE)
    mat[:3, -1] += np.array([dx, dy, dz], dtype=ARRAY_TYPE)
    return mat


def rot(theta, axis):
    ''' 齐次变换矩阵: 旋转'''
    theta = theta / 180 * np.pi
    # 角度 -> 弧度
    sin = np.sin(theta)
    cos = np.cos(theta)
    mat = np.eye(4)
    axis_idx = {'x': 0, 'y': 1, 'z': 2}
    if isinstance(axis, str):
        axis = axis_idx[axis]
        # 字符 -> 空间轴名称
    if axis == 0:
        mat[1: 3, 1: 3] = np.array([[cos, -sin],
                                    [sin, cos]], dtype=ARRAY_TYPE)
    elif axis == 1:
        mat[:3, :3] = np.array([[cos, 0, sin],
                                [0, 1, 0],
                                [-sin, 0, cos]], dtype=ARRAY_TYPE)
    elif axis == 2:
        mat[:2, :2] = np.array([[cos, -sin],
                                [sin, cos]], dtype=ARRAY_TYPE)
    else:
        raise AssertionError(f'axis: {axis_idx}')
    return mat


def conv_inv(conv):
    ''' 变换矩阵的逆'''
    *axes, pos = map(lambda i: conv[:3, i], range(4))
    inv = np.eye(4)
    inv[:3, :3] = conv[:3, :3].T
    for i, axis in enumerate(axes):
        inv[i, -1] = - (pos * axis).sum()
    return inv


if __name__ == '__main__':
    print(rot(30, 2))
