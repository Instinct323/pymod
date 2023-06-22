import numpy as np

from .basic import num, comentropy


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
    # 拷贝张量
    for idx, flag in enumerate(flags):
        col = data[:, idx]
        if isinstance(flag, bool):
            if not flag:
                data[:, idx] = col.max() - col
                # 成本型指标
        elif isinstance(flag, num):
            col = np.abs(col - flag)
            data[:, idx] = 1 - col / col.max()
            # 中间型指标
        elif len(flag) == 2:
            left, right = sorted(flag)
            if isinstance(left, num) and isinstance(right, num):
                col = (left - col) * (col < left) + (col - right) * (col > right)
                data[:, idx] = 1 - col / col.max()
                # 区间型指标
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
    if np.all(weight is None):
        length = pos.shape[1]
        weight = np.ones([1, length])
        # 当无权值要求，则各个指标权值相等
    else:
        weight = np.array(weight).reshape([1, -1])
        # 使用指定的权值
    weight /= weight.sum()
    # 令权值和为1
    worst = pos.min(axis=0)
    best = pos.max(axis=0)
    # 劣样本、优样本
    dis_p = ((weight * (pos - best)) ** 2).sum(axis=1) ** 0.5
    dis_n = ((weight * (pos - worst)) ** 2).sum(axis=1) ** 0.5
    # 样本到劣样本、优样本的距离
    score = dis_n / (dis_p + dis_n)
    # 计算得分
    return score.reshape(-1, 1)