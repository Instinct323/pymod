import logging

import matplotlib.pyplot as plt
import wmi

red = 'orangered'
orange = 'orange'
yellow = 'yellow'
green = 'greenyellow'
cyan = 'aqua'
blue = 'deepskyblue'
purple = 'mediumpurple'
pink = 'violet'
rainbow = [red, orange, yellow, green, blue, purple, pink]
# matplotlib 颜色常量

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def timer(repeat=1, avg=True):
    import time
    repeat = max([1, int(repeat) if isinstance(repeat, float) else repeat])

    def decorator(func):
        def handler(*args, **kwargs):
            start = time.time()
            result = tuple(func(*args, **kwargs) for i in range(repeat) if not i)[0]
            cost = (time.time() - start) * 1e3
            print(f'{func.__name__}: {cost / repeat if avg else cost:.3f} ms')
            return result

        return handler

    return decorator


def singleton(cls):
    ''' 单例类装饰器'''
    _instance = {}

    def _singleton(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton


def run_once(func):
    def handler(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as reason:
                print(reason)
                time.sleep(20)
                continue

    return handler


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            print(error)
            return error

    return handler


def set_brightness(value):
    ''' 调节屏幕亮度: value ∈ [0, 100]'''
    # import wmi
    connect = wmi.WMI(namespace='root/WMI')
    method = connect.WmiMonitorBrightnessMethods()[0]
    method.WmiSetBrightness(value, Timeout=500)


if __name__ == '__main__':
    for idx in range(108):
        print(f'\033[{idx}m{idx}')
