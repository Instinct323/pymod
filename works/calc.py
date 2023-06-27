import numpy as np


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
    import numpy as np
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

