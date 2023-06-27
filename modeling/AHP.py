import numpy as np

from .utils import cal_RI, solve_weight, LOGGER


def pair_comp_array(tril):
    ''' tril: 下三角矩阵 (以列为单元)
        return: 成对比较矩阵'''
    dim = len(tril)
    # 将下三角矩阵用 0 填充成方阵
    for row in range(dim):
        size = row + 1
        assert len(tril[row]) == size, f'第 {size} 行数据不满足下三角矩阵要求'
        tril[row].extend([0 for _ in range(dim - size)])
    array = np.array(tril)
    # 以成对比较矩阵的标准设置矩阵值
    for col in range(dim):
        for row in range(col):
            array[row][col] = 1 / array[col][row]
    return array


def solve_criterion_feature(array):
    ''' 求解准则层特征向量'''
    feature, CI = solve_weight(array)
    dim = feature.size
    # dim: 准则层指标数
    consistency_check(CI, dim, single=True)
    return feature


def solve_decision_weight(decision_array, criterion_feature):
    ''' 求解决策层权值矩阵
        decisions_array: 决策层成对比较矩阵序列
        criterion_feature: 准则层权向量'''
    weight_list = []
    CI_list = []
    # 分别求解决策层的各个权值向量
    for array in decision_array:
        weight, CI = solve_weight(array)
        weight_list.append(weight)
        CI_list.append(CI)
    # 拼接得到决策层权值矩阵
    decision_weight = np.stack(weight_list, axis=1)
    CI = (np.array(CI_list) * criterion_feature).sum()
    dim = decision_weight.shape[0]
    # dim: 决策层方案数
    consistency_check(CI, dim, single=False)
    return decision_weight


def consistency_check(CI, dim, single=True, decimals=5):
    ''' 一致性检验
        single: 是否为单排序一致性检验
        decimals: 数值精度
        CI = (λ - n) / (n - 1)
        CR = CI / RI
        Success: CR < 0.1'''
    if dim >= 3:
        RI = cal_RI(dim)
        CR = round(CI / RI, decimals)
        message = f'CI = {round(CI, decimals)}, RI = {RI}, CR = {CR}'
        success = CR < 0.1
    else:
        message = 'dim <= 2'
        success = True
    # 依照不同模式输出字符串
    head = 'Single sort consistency check' if single else 'Total sorting consistency check'
    LOGGER.info(f'{head}\nMessage: {message}\n')
    assert success, f'{head}: CR >= 0.1'
