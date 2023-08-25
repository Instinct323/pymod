import logging

import numpy as np

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)
np.set_printoptions(precision=3, suppress=True)
# debugger: 输出变量的信息
printvar = lambda *args: tuple(print(f'{x} = {eval(x)}') for x in args)


def main():
    # note: 固定的返回值 (局部变量)
    return locals()


if __name__ == '__main__':
    # note: main 函数中的 局部变量 加入 全局变量
    globals().update(main())
