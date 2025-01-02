from typing import Tuple, Union


def matrix(vector: Union[tuple, list],
           shape: Tuple[int, int],
           bracket: str = "[",
           newline: str = "\n") -> str:
    """ 将给定的向量转换为 LaTeX 格式的矩阵字符串
        :param vector: 向量
        :param shape: 矩阵的形状
        :param bracket: 矩阵的括号类型
        :param newline: 换行符
        :return LaTeX 格式的矩阵字符串"""
    h, w = shape
    assert len(vector) == h * w, f"shape {shape} mismatched with vector {len(vector)}"
    array = [vector[i * w: (i + 1) * w] for i in range(h)]
    env = {"": "", "(": "p", "[": "b", "{": "B"}[bracket] + "matrix"

    ret = [f"\\begin{{{env}}}"]
    for row in array:
        ret.append(" & ".join(map(str, row)) + r" \\")
    ret.append(f"\\end{{{env}}}")
    return newline.join(ret)


def cases(rows: Union[tuple, list],
          newline: str = "\n") -> str:
    ret = [r"\begin{cases}"]
    for r in rows:
        ret.append((" & ".join(map(str, r)) if isinstance(r, (tuple, list)) else str(r)) + r" \\")
    ret.append(r"\end{cases}")
    return newline.join(ret)


if __name__ == '__main__':
    print(matrix(["K_p", "K_i", "K_d"], (1, 3)))
