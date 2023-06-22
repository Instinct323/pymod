import numpy as np


def eval_weight(R):
    ''' 评价指标权重向量A:
        变异系数法 求 评价指标权重 (反映指标分辨能力)'''
    mean = R.mean(axis=0, keepdims=True)
    variance = np.sqrt(((R - mean) ** 2).sum(axis=0, keepdims=True) / (R.shape[0] - 1))
    weight = variance / mean
    weight /= weight.sum()
    return weight


def rela_dev_fun(data, is_benefit):
    ''' 相对偏差法:
        data:
            行索引: 各方案
            列索引: 各指标
        return: 模糊综合评价矩阵R'''
    plan = []
    for col, flag in zip(data.T, is_benefit):
        if flag:
            choose = col.max()
        else:
            choose = col.min()
        plan.append(choose)
    plan = np.array(plan)
    print(f'虚拟理想方案: {plan}')
    array = np.abs(plan - data) / (data.max(axis=0) - data.min(axis=0))
    return array


def rela_opti_fun(data, is_benefit):
    ''' 相对优属度法:
        data:
            行索引: 各方案
            列索引: 各指标
        return: 模糊综合评价矩阵R'''
    for idx, flag in enumerate(is_benefit):
        col = data[:, idx]
        if flag:
            data[:, idx] /= col.max()
        else:
            data[:, idx] = col.min() / col
    return data


def fuzzy_oper(A, R, mode):
    ''' mode 模糊算子模式
        {0: (&, |), 1: (*, |), 2: (&, +), 3: (*, +)}'''
    assert 0 <= mode <= 3
    A = A.reshape(-1, 1)
    if mode % 2:
        result = A * R
    else:
        result = np.fmin(A, R)
    if mode < 2:
        result = result.max(axis=0)
    else:
        result = result.sum(axis=0)
    result /= result.sum()
    return result