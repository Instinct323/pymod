import sympy


def Solve_Ode(ode, symbol, diff_time, subject_to=None, known_symbols=None):
    ''' :param ode: n阶常微分方程
        :param diff_time: 微分次数
        :param symbol: 自变量 [symbol, value]
        :param subject_to: 约束条件 {0: {x(0): 1, …}}
        :param known_symbols: 已知变量(set)'''
    if subject_to is None:
        subject_to = {}
    if known_symbols is None:
        known_symbols = set({})
    solution = sympy.dsolve(ode)
    ode = solution.rhs - solution.lhs
    free_symbols = ode.free_symbols - known_symbols
    equation_set = []
    for value in subject_to:
        condition = subject_to[value]
        for n in range(diff_time + 1):
            equation = ode.diff(symbol, n).subs(symbol, value).subs(condition)
            equation_set.append(equation)
    free_symbols = sympy.solve(equation_set, free_symbols)
    result = solution.subs(free_symbols)
    return result
