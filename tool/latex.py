import re
from typing import Tuple, Union, Callable

import numpy as np


def vec2latexmat(vector: Union[tuple, list],
                 shape: Tuple[int, int],
                 bracket: str = "[",
                 newline: str = "\n") -> str:
    """ 将给定的向量转换为 LaTeX 格式的矩阵字符串
            tips: 类似的用法还有 aligned, cases
        :param vector: 向量
        :param shape: 矩阵的形状
        :param bracket: 矩阵的括号类型
        :param newline: 换行符
        :return LaTeX 格式的矩阵字符串 """
    h, w = shape
    assert len(vector) == h * w, f"shape {shape} mismatched with vector {len(vector)}"
    array = [vector[i * w: (i + 1) * w] for i in range(h)]
    env = {"": "", "(": "p", "[": "b", "{": "B"}[bracket] + "matrix"

    ret = [f"\\begin{{{env}}}"]
    for row in array:
        ret.append(" & ".join(map(str, row)) + r" \\")
    ret.append(f"\\end{{{env}}}")
    return newline.join(ret)


def from_numpy(matrix: np.ndarray, **kwargs) -> str:
    return vec2latexmat(matrix.flatten(), shape=matrix.shape, **kwargs)


def from_latexmat(context: str,
                  transform: Callable = None) -> np.ndarray:
    """ latex 矩阵字符串转换为 numpy 数组
        :param context: LaTeX 格式的矩阵字符串
        :return numpy 数组 """
    if transform is None: transform = lambda x: x
    context = [list(map(transform, row.split("&")))
               for row in re.sub(r"\s+", "", context).split(r"\\") if row]
    return np.array(context)


if __name__ == '__main__':
    # print(vec2latexmat([""] * 9, (3, 3)))
    P = from_latexmat(r"""0 & 1 & 10 & -3 \\ 
0 & 2 & 9 & -4 \\
0 & 3 & 8 & -5 \\""", float)
    P_ = from_latexmat(r"""-3 & 10 & 1 & 0 \\
-4 & 9 & 2 & 0 \\
-5 & 8 & 3 & 0 \\""", float)

    Q = P - P.mean(axis=-1, keepdims=True)
    Q_ = P_ - P_.mean(axis=-1, keepdims=True)
    print("Q_ =", from_numpy(Q_))

    W = Q @ Q_.T
    print("W =", from_numpy(W))

    u, s, vh = np.linalg.svd(W)
    R = u @ vh
    print("R =", from_numpy(np.round(R, 4)))

    t = np.mean(P - R @ P_, axis=-1, keepdims=True)
    print("t =", from_numpy(np.round(t, 4)))

    RP_t = R @ P_ + t
    error = P - RP_t
    print(np.square(error).sum())
