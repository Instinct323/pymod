import numpy as np

from .utils import NUMBER, comentropy


def positive(data, flags):
    ''' data:
            行索引: 各方案
            列索引: 各指标
        flags: 指标类型
            True: 效益型
            False: 成本型
            num: 中间型
            [num, num]: 区间型
        return: 正向化, 标准化矩阵'''
    data = data.copy()
    for i, flag in enumerate(flags):
        col = data[:, i]
        # 成本型指标
        if isinstance(flag, bool):
            if not flag:
                data[:, i] = col.max() - col
        # 中间型指标
        elif isinstance(flag, NUMBER):
            col = np.abs(col - flag)
            data[:, i] = 1 - col / col.max()
        # 区间型指标
        elif len(flag) == 2:
            l, r = sorted(flag)
            if isinstance(l, NUMBER) and isinstance(r, NUMBER):
                col = (l - col) * (col < l) + (col - r) * (col > r)
                data[:, i] = 1 - col / col.max()
            else:
                raise AssertionError('区间型指标数据类型出错')
        else:
            raise AssertionError('出现无法识别的指标类型')
    data /= (data ** 2).sum(axis=0) ** 0.5
    return data


def cal_score(pos, weight=None):
    ''' pos: 正向化, 标准化矩阵
        weight: 权重向量
        return: 样本得分'''
    # 当无权值要求，则各个指标权值相等
    if np.all(weight is None):
        length = pos.shape[1]
        weight = np.ones([1, length])
    # 使用指定的权值
    else:
        weight = np.array(weight).reshape([1, -1])
    weight /= weight.sum()
    # 样本到劣样本、优样本的距离
    dis_p = ((weight * (pos - pos.max(axis=0))) ** 2).sum(axis=1) ** 0.5
    dis_n = ((weight * (pos - pos.min(axis=0))) ** 2).sum(axis=1) ** 0.5
    return (dis_n / (dis_p + dis_n)).reshape(-1, 1)
