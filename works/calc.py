import numpy as np


def seq_comp(seq1, seq2):
    ''' 序列匹配度计算'''
    n1, n2 = map(len, (seq1, seq2))
    dp = [[int(e1 == e2) for e2 in seq2] for e1 in seq1]
    for c in range(1, n2): dp[0][c] = max(dp[0][c], dp[0][c - 1])
    for r in range(1, n1):
        dp[r][0] = max(dp[r][0], dp[r - 1][0])
        for c in range(1, n2):
            dp[r][c] = dp[r - 1][c - 1] + 1 \
                if dp[r][c] else max(dp[r - 1][c], dp[r][c - 1])
    return dp[-1][-1] / max(n1, n2)


def salary(pre_tax, Si=485.12, Hf=230):
    ''' pre_tax: 税前薪资 (k)
        Si: 社会保险
        Hf: 住房公积金'''
    result = pre_tax - (Si + Hf) / 1000
    # 应纳税所得额
    taxable_income = np.maximum(0, result - 5)
    # 个人所得税税率表
    iit_dict = ((0, 36, 0.03), (36, 144, 0.1), (144, 300, 0.2), (300, 420, 0.25),
                (420, 660, 0.3), (660, 960, 0.35), (960, float('inf'), 0.45))
    result -= sum(np.maximum(0, np.minimum(taxable_income, t) - b) * rate for b, t, rate in iit_dict)
    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    pre_tax = np.linspace(11, 17, 100)
    after_tex = salary(pre_tax)
    diff = pre_tax - after_tex

    plt.xlabel('税前 (k)')
    plt.ylabel('差额 (k)')
    plt.plot(pre_tax, diff, color='deepskyblue')
    plt.grid()
    plt.show()
