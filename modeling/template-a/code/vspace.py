import numpy as np


# debugger: 输出变量的信息
def printvar(*args):
    for x in args: print(f'{x} = {eval(x)}')


l = 0.6  # 初始吃水深度
lmax = 2
lmin = 0

vw = 24  # 风速
rb = 1  # 浮标半径
rc = 0.025  # 钢管半径
rd = 0.15  # 钢桶半径
ld = 1  # 钢桶长度
lc = 1  # 钢管长度
la = 22.05  # 锚链长度
lh = 18  # 水深
mq = 1200  # 配重球
mc = 10  # 钢管质量
md = 100  # 钢桶质量
mb = 1000  # 浮标质量
g = 9.8  # 重力加速度
sig = 7  # 单位长度的钢管质量
lb = 2  # 浮标总高
lt = 0  # 拖地长度
ps = 1025  # 海水密度
Gq = -mq * g
Gc = -mc * g
Gd = -md * g
Gb = -mb * g
Gs = -sig * (la - lt) * g  # 锚链重力
Fcy = ps * g * np.pi * lc * rc ** 2
Fdy = ps * g * np.pi * ld * rd ** 2

endtime = 100


def y(x, H, C1, sig, g):
    H = abs(H)
    return H / (sig * g) * np.cosh((sig * g) / H * x) - H / (sig * g)


def main():
    # note: 如果对全局变量进行修改, 在此声明
    global l, lmax, lmin

    for _ in range(endtime):
        Fby = ps * g * np.pi * l * rb ** 2

        F6y = 4 * Gc + Gd + Gq + Gb + Gs + (Fby + 4 * Fcy + Fdy)
        F5y = F6y + Gq + Gs
        F4y = F5y + (Fdy + Gd)
        F3y = F4y + (Fcy + Gc)
        F2y = F4y + 2 * (Fcy + Gc)
        F1y = F4y + 3 * (Fcy + Gc)
        F0y = F4y + 4 * (Fcy + Gc)
        F0yy = Fby + Gb

        H0x = 0.625 * vw ** 2 * 2 * rb * (lb - l)
        H = -H0x
        C1 = np.arcsinh(-(F6y / H)) * (H / (sig * g))

        # a = H / (sig * g) * np.arcsinh(((sig * g) / H * (la - lt) + np.arcsinh(C1)) - C1) + lt
        a = -H / (sig * g) * np.arcsinh(-(sig * g) / H * (la - lt)) - C1

        b1 = np.arctan((F1y + F0y) / (2 * H))
        b2 = np.arctan((F2y + F1y) / (2 * H))
        b3 = np.arctan((F3y + F3y) / (2 * H))
        b4 = np.arctan((F4y + F3y) / (2 * H))
        b5 = np.arctan((F5y + F4y) / (2 * H))

        # 系统总高 - 水深
        pd1 = y(a, H, C1, sig, g) + l + lc * (np.sin(b1) + np.sin(b2) + np.sin(b3) + np.sin(b4)) + ld * np.sin(b5) - lh

        # 算出来总高超过了海床，吃水过多
        if pd1 > 0:
            lmax = l
        else:
            lmin = l
        l = (lmax + lmin) / 2

    R = lt + a + lc * (np.cos(b1) + np.cos(b2) + np.cos(b3) + np.cos(b4)) + ld * np.cos(b5)

    # note: 固定的返回值
    return locals()


if __name__ == '__main__':
    # note: main 函数中的 局部变量 加入 全局变量
    globals().update(main())

    printvar(
        'R',
        'l',
        'lt',
        'a',
        'C1',
        'b1',
        'b2',
        'b3',
        'b4',
        'b5',
        'F6y',
        'y(0, H, C1, sig, g)',
        'y(2, H, C1, sig, g)',
        'y(4, H, C1, sig, g)',
        'y(8, H, C1, sig, g)',
        'y(16, H, C1, sig, g)',
        'y(a, H, C1, sig, g)'
    )
