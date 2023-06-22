from math import factorial

import numpy as np
import sympy


def poisson_distribution(var, mean):
    ''' 泊松分布'''
    assert var.dtype == np.int, '变量需为 int'
    fact = np.array(list(map(factorial, var)))
    return mean ** var / fact * np.exp(-mean)


def update_state(basic, proba, time):
    ''' basic: 当前状态
        proba: 转移概率矩阵
        time: 转移次数
        return: 新状态概率分布'''
    return np.linalg.multi_dot([basic] + [proba] * time)


def steady_state(proba):
    ''' proba: 转移概率矩阵
        return: 马氏链稳态概率分布'''
    assert len(proba) == len(proba[0]), '转移概率矩阵应为方阵'
    for row in proba:
        assert sum(row) == 1, '不满足转移概率矩阵条件'
    dim = proba.shape[0]
    coef = (proba - np.eye(dim)).T
    # 解: (P - E)x = 0
    zero_space = np.array(sympy.Matrix(coef).rref()[0]).T
    # 得到零空间
    x = np.array(sympy.symbols([f'x{idx}' for idx in range(dim)]))
    gen_sol = sympy.solve(x @ zero_space, *x)
    # 求得通解
    part_sol = sympy.solve([sum(x) - 1] + [key - value for key, value in gen_sol.items()], *x)
    # 求得特解: 和为 1 条件下
    result = np.array([part_sol[sym] for sym in x])
    return result


if __name__ == '__main__':
    print(steady_state(np.array([[0.3679, 0., 0.6321],
                                 [0.3679, 0.3679, 0.2642],
                                 [0.1839, 0.3679, 0.4482]])))