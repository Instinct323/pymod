from typing import Tuple, Union


def matrix(vector: Union[tuple, list],
           shape: Tuple[int, int],
           bracket: str = "[",
           sep="\n") -> str:
    h, w = shape
    assert len(vector) == h * w, f"shape {shape} mismatched with vector {len(vector)}"
    array = [vector[i * w: (i + 1) * w] for i in range(h)]
    env = {"": "", "(": "p", "[": "b", "{": "B"}[bracket] + "matrix"

    ret = [f"\\begin{{{env}}}"]
    for row in array:
        ret.append(" & ".join(map(str, row)) + r" \\")
    ret.append(f"\\end{{{env}}}")
    return sep.join(ret)


def cases(rows: Union[tuple, list], sep="\n") -> str:
    ret = [r"\begin{cases}"]
    for r in rows:
        ret.append(str(r) + r" \\")
    ret.append(r"\end{cases}")
    return sep.join(ret)


if __name__ == '__main__':
    print(matrix([r"\cos(-\theta)", r"\sin(-\theta)", r"\sin(-\theta)", r"\cos(-\theta)"], (2, 2), bracket="["))
