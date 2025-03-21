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
    A = from_latexmat(r"""11 & 0.603975 \\
0.603975 & 0.062321""", float)
    b = from_latexmat(r"""13.639649 \\ 0.530331""", float)
    print(np.linalg.solve(A, b))

