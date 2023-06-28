import warnings

import numpy as np
import pandas as pd
from scipy import optimize

warnings.filterwarnings('ignore')


def is_success(result):
    # 检测规划结果是否成功
    if not result.success:
        raise ValueError(result.message)
    return True


def np2mat(array, reshape=False):
    ''' 将 numpy 转化成 matrix'''
    if np.any(array is not None):
        array = array.astype(np.float)
        if reshape: array = array.reshape(-1, 1)
        return matrix(array)


class Result:
    ''' Result:
        解决 cvxopt 与 scipy 的兼容问题'''
    success = True

    def __init__(self, result):
        self.x = np.array(result['x']).reshape(-1)
        self.fun = result['primal objective']


class Programme:
    ''' 规划问题模型
        .data: 最优解表格
        .x: 最优浮点数解
        .int_num: 整数规划的x数目
        .show_all: 是否筛除浮点数解'''
    min = 'min'
    data = None
    x = None

    def __init__(self, num, bounds, int_num, show_all):
        self.num = num
        self.int_num = int_num if int_num <= num else num
        self.int_mask = np.array([1 for _ in range(self.int_num)] + [0 for _ in range(self.num - self.int_num)])
        self.bounds = (bounds,) * num
        self.show_all = show_all

    def row_data(self, result, decimals):
        solution = dict(enumerate(np.round(result.x, decimals)))
        solution[self.min] = result.fun
        return solution

    def beautify(self, columns):
        self.data[1:] = self.data[1:].sort_values(by=self.min)
        if columns:
            columns.append(self.min)
            self.data.columns = columns
            self.data = self.data.drop_duplicates(keep='first', ignore_index=True)

    def init_data(self, result, decimals):
        solution = self.row_data(result, decimals=decimals)
        return pd.DataFrame(solution, index=[0])

    def int_prog(self, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
        if not self.check_int(self.x):
            ceil, floor = - np.ceil(x) * self.int_mask, np.floor(x) * self.int_mask
            for idx in range(self.int_num):
                for bound, w in ((ceil, -1), (floor, 1)):
                    new_A_ub = np.zeros([1, self.num])
                    new_A_ub[0, idx] = w
                    new_b_ub = bound[idx].reshape([1, 1])
                    if np.any(A_ub is not None):
                        new_A_ub = np.concatenate([A_ub, new_A_ub], axis=0)
                        new_b_ub = np.concatenate([b_ub, new_b_ub], axis=1)
                    yield new_A_ub, new_b_ub, A_eq, b_eq

    def check_int(self, x):
        return (np.abs(x - np.round(x, 0)) * self.int_mask).sum() < 1e-4

    def handle_branch(self, branch, decimals):
        if branch.success:
            x = branch.x
            if self.show_all or self.check_int(x):
                solution = self.row_data(branch, decimals)
                self.data = self.data.append(solution, ignore_index=True)

    def __repr__(self):
        return self.data


class Linprog(Programme):
    ''' 线性规划
        引入正松弛变量: 不等式 => 等式
        互斥约束问题: 决策变量y, 4x+5y < 200+(1-y)*M, M -> ∞
        condition: A_ub @ x - b_ub <= 0, A_eq @ x - b_eq == 0
        int_num: 整数规划的变量数
        decimals: 结果精度'''

    def __init__(self, weight, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=(0, None),
                 columns=None, decimals=3, int_num=0, show_all=False):
        super().__init__(weight.size, bounds=bounds, int_num=int_num, show_all=show_all)
        result = optimize.linprog(weight, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        if is_success(result):
            self.data = self.init_data(result, decimals)
            self.x = result.x
            if self.int_num:
                for A_ub, b_ub, A_eq, b_eq in self.int_prog(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq):
                    branch = optimize.linprog(weight, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
                    self.handle_branch(branch, decimals)
            self.beautify(columns)


class Non_Linprog(Programme):
    ''' 非线性规划
        fun: 需要优化的函数
        x0: 搜索的初态
        condition: A_ub @ x - b_ub <= 0, A_eq @ x - b_eq == 0
        int_num: 整数规划的变量数
        decimals: 结果精度'''
    methods = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
               'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr',
               'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    bounds = None

    def __init__(self, fun, x0, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=(0, None),
                 cond_list=None, method=8,
                 columns=None, decimals=3, int_num=0, show_all=False):
        super().__init__(x0.size, bounds=bounds, int_num=int_num, show_all=show_all)
        self.constraints = None
        self.init_cond(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, cond_list=cond_list)
        result = optimize.minimize(fun, x0, method=self.methods[method], bounds=self.bounds,
                                   constraints=self.constraints)
        if is_success(result):
            self.data = self.init_data(result, decimals)
            self.x = result.x
            if self.int_num:
                for A_ub, b_ub, A_eq, b_eq in self.int_prog(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq):
                    self.init_cond(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, cond_list=cond_list)
                    branch = optimize.minimize(fun, x0, method=self.methods[method], bounds=self.bounds,
                                               constraints=self.constraints)
                    self.handle_branch(branch, decimals)
            self.beautify(columns)

    def init_cond(self, A_ub=None, b_ub=None, A_eq=None, b_eq=None, cond_list=None):
        self.constraints = []
        self.lin_cond(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        self.non_lin_cond(cond_list)

    def lin_cond(self, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
        if np.any(A_ub is not None) and np.any(b_ub is not None):
            b_ub = b_ub.reshape(-1)
            self.add_cond(True, lambda x: b_ub - x @ A_ub.T)
        if np.any(A_eq is not None) and np.any(b_eq is not None):
            b_eq = b_eq.reshape(-1)
            self.add_cond(False, lambda x: x @ A_eq.T - b_eq)

    def non_lin_cond(self, cond_list=None):
        if cond_list:
            for operator, fun in cond_list:
                if operator == '>=':
                    self.add_cond(True, fun)
                elif operator == '==':
                    self.add_cond(False, fun)
                else:
                    raise ValueError('无法识别\'>=\'、\'==\'以外的运算符')

    def add_cond(self, ineq, fun):
        self.constraints.append({'type': 'ineq' if ineq else 'eq', 'fun': fun})


class Quad_Prog(Programme):
    ''' 二次规划'''

    def __init__(self, quad, linear, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                 columns=None, decimals=3, int_num=0, show_all=False):
        from cvxopt.solvers import qp
        super().__init__(linear.size, bounds=(0, None), int_num=int_num, show_all=show_all)
        quad = np2mat(quad)
        linear = np2mat(linear, reshape=True)
        result = qp(P=quad, q=linear, G=np2mat(A_ub), h=np2mat(b_ub, reshape=True),
                    A=np2mat(A_eq), b=np2mat(b_eq, reshape=True))
        result = Result(result)
        self.data = self.init_data(result, decimals)
        self.x = result.x
        if self.int_num:
            int_prog = self.int_prog(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
            for A_ub, b_ub, A_eq, b_eq in int_prog:
                branch = qp(P=quad, q=linear, G=np2mat(A_ub), h=np2mat(b_ub, reshape=True),
                            A=np2mat(A_eq), b=np2mat(b_eq, reshape=True))
                branch = Result(branch)
                self.handle_branch(branch, decimals)
        self.beautify(columns)


if __name__ == '__main__':
    pass
