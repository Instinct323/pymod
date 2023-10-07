import numpy as np


def eval_weight(R):
    ''' 评价指标权重向量A:
        变异系数法 求 评价指标权重 (反映指标分辨能力)'''
    mean = R.mean(axis=0, keepdims=True)
    variance = np.sqrt(((R - mean) ** 2).sum(axis=0, keepdims=True) / (R.shape[0] - 1))
    weight = variance / mean
    return weight / weight.sum()


def rela_dev_fun(data, is_benefit):
    ''' 相对偏差法:
        :return: 模糊综合评价矩阵R'''
    plan = np.array([col.max() if flag else col.min() for col, flag in zip(data.T, is_benefit)])
    print(f'虚拟理想方案: {plan}')
    return np.abs(plan - data) / (data.max(axis=0) - data.min(axis=0))


def rela_opti_fun(data, is_benefit):
    ''' 相对优属度法:
        :return: 模糊综合评价矩阵R'''
    for i, flag in enumerate(is_benefit):
        col = data[:, i]
        data[:, i] = col / col.max() if flag else col.min() / col
    return data


def fuzzy_oper(A, R, mode):
    ''' :param mode: 模糊算子模式
            {0: (&, |), 1: (*, |), 2: (&, +), 3: (*, +)}'''
    assert 0 <= mode <= 3
    A = A.reshape(-1, 1)
    result = A * R if mode % 2 else np.fmin(A, R)
    result = result.max(axis=0) if mode < 2 else result.sum(axis=0)
    return result / result.sum()
