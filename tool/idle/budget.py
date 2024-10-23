import numpy as np
import pandas as pd

str2date = lambda s: pd.PeriodIndex(s, freq="M")
# 计算的起始日期
start, end = str2date(["2024-03", "2027-12"])
# 当前存款, 每月开销
fund = 3.4
livcost = 1.5

MONTHLY = pd.read_excel(r"D:\Information\Lib\月度收支.xlsx")
for k in ("start", "end"): MONTHLY[k] = str2date(MONTHLY[k])


def surplus(show=False):
    ret = pd.Series(- livcost, index=pd.period_range(start, end))
    ret[0] += fund
    for i, (s, e, v, detail) in MONTHLY.iterrows():
        ret += ((ret.index >= s) & (ret.index <= e)) * float(v)

    if show:
        plt.plot(ret.cumsum().to_list(), color="deepskyblue")
        plt.xticks(np.arange(len(ret)), ret.index, rotation=90)
        plt.ylabel("fund (k)")
        plt.grid(), plt.tight_layout(), plt.show()
    return ret


def prepare(*plans):
    for s, e, total, detail in sorted(plans):
        tmp = surplus()[s:]
        s, e = str2date([s, e])
        # 计算最快筹备月, 逐月转账
        total = min(total, tmp.cumsum().max())
        e = tmp.index[(tmp.cumsum() >= total).argmax()]
        tmp = tmp[:e]
        tmp[e] = total - tmp[s:e - 1].sum()
        # 合并相同的转账
        diff = tmp.diff()
        while e <= tmp.index[-1]:
            e = s + 1
            while e < tmp.index[-1] and diff[e] == 0: e += 1
            # 插入支出项目
            MONTHLY.loc[len(MONTHLY)] = s, e - 1, -tmp[s], detail
            s = e


def salary(pre_tax, Si=485.12, Hf=230):
    """ :param pre_tax: 税前薪资 (k)
        :param Si: 社会保险
        :param Hf: 住房公积金"""
    result = pre_tax - (Si + Hf) / 1000
    # 应纳税所得额
    taxable_income = np.maximum(0, result - 5)
    # 个人所得税税率表
    iit_dict = ((0, 36, 0.03), (36, 144, 0.1), (144, 300, 0.2), (300, 420, 0.25),
                (420, 660, 0.3), (660, 960, 0.35), (960, float("inf"), 0.45))
    result -= sum(np.maximum(0, np.minimum(taxable_income, t) - b) * rate for b, t, rate in iit_dict)
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    prepare(
        # ("2024-03", "2024-08", 12.0, "学费, 生活费, 宿舍用品, 备用资金"),
        ("2024-03", None, 48.0, "还贷")
    )

    print(MONTHLY.sort_values(by="start").reset_index(drop=True))
    print(surplus(show=True))
