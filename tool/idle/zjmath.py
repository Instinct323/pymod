import numpy as np
import sympy as sp


def eig_poly(arr):
    """ 特征多项式的系数"""
    lam = sp.symbols("s")
    coef = np.zeros([len(arr) + 1])
    arr = sp.eye(len(arr)) * lam - sp.Matrix(arr)
    poly = arr.det()
    for item in poly.args:
        item = sorted(item.args, key=lambda s: "s" not in str(s)) \
            if isinstance(item, sp.Mul) else [item]
        # 常数项系数
        if len(item) == 1 and "s" not in str(item[0]):
            coef[0] += item[0]
            continue
        # 其它项系数
        s_i, *a_i = item
        power = int(s_i.args[1] if s_i.args else 1)
        a_i = a_i[0] if a_i else 1
        coef[power] += a_i
    return coef


def seq_comp(seq1, seq2):
    """ 序列匹配度计算"""
    n1, n2 = map(len, (seq1, seq2))
    dp = [[int(e1 == e2) for e2 in seq2] for e1 in seq1]
    for c in range(1, n2): dp[0][c] = max(dp[0][c], dp[0][c - 1])
    for r in range(1, n1):
        dp[r][0] = max(dp[r][0], dp[r - 1][0])
        for c in range(1, n2):
            dp[r][c] = dp[r - 1][c - 1] + 1 \
                if dp[r][c] else max(dp[r - 1][c], dp[r][c - 1])
    return dp[-1][-1] / max(n1, n2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

    pre_tax = np.linspace(11, 17, 100)
    after_tex = salary(pre_tax)
    diff = pre_tax - after_tex

    plt.xlabel("税前 (k)")
    plt.ylabel("差额 (k)")
    plt.plot(pre_tax, diff, color="deepskyblue")
    plt.grid()
    plt.show()
