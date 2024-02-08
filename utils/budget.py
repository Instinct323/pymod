import numpy as np
import pandas as pd

str2date = lambda s: pd.PeriodIndex(s, freq='M')
# 计算的起始日期
start, end = str2date(['2024-03', '2027-12'])
# 当前存款, 每月开销
fund = 3.4
livcost = 1.5

MONTHLY = pd.read_excel(r'D:\Information\Lib\月度收支.xlsx')
for k in ('start', 'end'): MONTHLY[k] = str2date(MONTHLY[k])


def surplus(show=False):
    ret = pd.Series(- livcost, index=pd.period_range(start, end))
    ret[0] += fund
    for i, (s, e, v, detail) in MONTHLY.iterrows():
        ret += ((ret.index >= s) & (ret.index <= e)) * float(v)

    if show:
        plt.plot(ret.cumsum().to_list(), color='deepskyblue')
        plt.xticks(np.arange(len(ret)), ret.index, rotation=90)
        plt.ylabel('fund (k)')
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    score = 3.554 / 5 * 30 + 80 * 0.3
    # 国赛: 蓝桥 * 3, 2023 数模
    # 省赛: 2022 数模
    # 校赛: 2022 数模, 2022 众盈杯
    # 院赛: 2022 计算机仿真, 2022 大数据应用
    score += 0.4 * (10 * 4 + 5 * 1 + 3 * 4)
    print(f'南科大奖学金分数: {score}')

    prepare(
        # ('2024-03', '2024-08', 12.0, '学费, 生活费, 宿舍用品, 备用资金'),
        ('2024-03', None, 48.0, '还贷')
    )

    print(MONTHLY.sort_values(by='start').reset_index(drop=True))
    print(surplus(show=True))
