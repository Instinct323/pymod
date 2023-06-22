import numpy as np
import sympy as sp


def eig_poly(arr):
    ''' 特征多项式的系数'''
    lam = sp.symbols('s')
    coef = np.zeros([len(arr) + 1])
    arr = sp.eye(len(arr)) * lam - sp.Matrix(arr)
    poly = arr.det()
    for item in poly.args:
        item = sorted(item.args, key=lambda s: 's' not in str(s)) \
            if isinstance(item, sp.Mul) else [item]
        # 常数项系数
        if len(item) == 1 and 's' not in str(item[0]):
            coef[0] += item[0]
            continue
        # 其它项系数
        s_i, *a_i = item
        power = int(s_i.args[1] if s_i.args else 1)
        a_i = a_i[0] if a_i else 1
        coef[power] += a_i
    return coef
