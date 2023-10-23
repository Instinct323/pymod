import numpy as np

from utils import *

str2date = lambda s: pd.PeriodIndex(s, freq='M')
# 计算的起始日期
start, end = str2date(['2024-03', '2027-12'])
# 当前存款, 每月开销
fund = 5.5
livcost = 1.5


def surplus(show=False):
    monthly = get_table('budget.monthly')
    for k in ('start', 'end'): monthly[k] = str2date(monthly[k])

    ret = pd.Series(- livcost, index=pd.period_range(start, end))
    ret[0] += fund
    for i, (s, e, v, detail) in monthly.iterrows():
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
        e = tmp.index[(tmp.cumsum() >= total).argmax()]
        tmp = tmp[:e]
        tmp[e] = total - tmp[s:e - 1].sum()
        # 合并相同的转账
        diff = tmp.diff()
        while e <= tmp.index[-1]:
            e = s + 1
            while e < tmp.index[-1] and diff[e] == 0: e += 1
            # 执行 SQL 语句, 插入支出项目
            CURSOR.execute(f"insert into budget.monthly values ('{s}', '{e - 1}', -{tmp[s]}, '{detail}');")
            s = e
        CONNECT.commit()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    if 0:
        prepare(
            ('2024-03', '2024-08', 14.0, '学费, 生活费, 宿舍用品, 备用资金'),
            ('2024-09', None, 48.0, '还贷')
        )

    fund = surplus(show=True).cumsum()
    print(fund)
