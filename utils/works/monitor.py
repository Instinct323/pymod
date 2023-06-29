import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import wmi

from mod.zjdl.utils.utils import Capture


def set_brightness(value):
    ''' 调节屏幕亮度: value ∈ [0, 100]'''
    # import wmi
    connect = wmi.WMI(namespace='root/WMI')
    method = connect.WmiMonitorBrightnessMethods()[0]
    method.WmiSetBrightness(value, Timeout=500)


class Monitor:
    ''' 动作捕捉监控
        time_interval: 记录图像的时间间隔
        sensitivity: 警报的回溯时间
        screen_ctrl: 屏幕控制
        log_path: 异常图像缓存目录'''
    time_interval = 1
    sensitivity = 10
    screen_ctrl = False
    log_path = Path('Log')
    # 高级常量
    _running_mean = 0.7
    _running_var = 0
    _warn_thresh = 1e-2

    def __init__(self):
        self.env()
        self.video = Capture()
        self._momentum = 1 - pow(0.1, self.time_interval / self.sensitivity)
        print(f'The momentum is set to {self._momentum:.3f}\n')
        print(('%10s' * 6) % ('Size (MB)', 'r_Mean', 'f_Mean', 'r_Var', 'f_Var', 'Stat'))
        # main
        for i, img in enumerate(self.video):
            start = time.time()
            # 保存监控图像
            float_mean, float_var, stat = self.guard(i + 1, img)
            # 其它信息管理
            size = self.get_size()
            print(('\r' + '%10.3f' * 5 + '%10s') % (size, self._running_mean, float_mean,
                                                    self._running_var, float_var, stat), end='')
            # 短暂休眠
            res_time = max([0, self.time_interval - (time.time() - start)])
            if res_time: time.sleep(res_time)

    def env(self):
        ''' 创建缓存目录'''
        for path in [self.log_path]:
            if path.is_dir(): shutil.rmtree(path)
            path.mkdir()
        if self.screen_ctrl: pass

    def get_size(self):
        ''' 获取占用空间信息'''
        size = sum([file.stat().st_size for file in self.log_path.iterdir()])
        return size / pow(2, 20)

    def guard(self, i, img):
        ''' 根据图像变化警报'''
        t = time.strftime('-'.join(['%H', '%M', '%S']), time.localtime())
        gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY) / 255
        mean, var = gray.mean(), gray.var()
        # 计算浮动值
        float_mean = abs(mean - self._running_mean)
        float_var = abs(var - self._running_var)
        # 判断当前状态
        stat = 'Init'
        if i >= self.sensitivity * 3:
            warning = float_mean + float_var > self._warn_thresh
            stat = 'Warn!' if warning else 'OK'
            # 警戒状态处理
            if warning:
                if self.screen_ctrl: set_brightness(0)
                time.sleep(self.time_interval / 3)
                cv2.imwrite(str(self.log_path / f'{t}.jpg'), next(self.video))
                if self.screen_ctrl: set_brightness(100)
        # 更新统计值
        update = lambda sta, cur: self._momentum * cur + (1 - self._momentum) * sta
        self._running_mean = update(self._running_mean, mean)
        self._running_var = update(self._running_var, var)
        return float_mean, float_var, stat


def SightSaver(env_range=[.06, .7],
               scr_range=[1.1, 1.8],
               interval: float = 3,
               stride: int = 5,
               ulimit: int = 100):
    ''' 屏幕亮度管理
        env_range: 启用自适应亮度的环境亮度区间
        scr_range: 画面亮度对眼睛的刺激程度
        interval: 刷新亮度的间隔
        stride: 屏幕亮度的步长
        ulimit: 屏幕亮度的上限'''
    mean, momentum = 1., .8
    # 使用 HSV 颜色空间定义的亮度
    get_mean = lambda img: img.max(axis=-1).astype(np.float16).mean() / 255
    env_range[1] -= env_range[0]
    scr_range[1] -= scr_range[0]
    # 环境、画面亮度转换函数
    f_env = lambda env: max([0, env - env_range[0]]) / env_range[1]
    g_scr = lambda scr: scr_range[0] + scr_range[1] * scr
    # 记录当前屏幕亮度
    cur_br, steps = ulimit, ulimit // stride
    for img in Capture(dpi=[640, 360]):
        env = get_mean(img[-180:])
        screen = get_mean(np.flip(pyautogui.screenshot(), axis=-1))
        # f(env) = radio * g(screen)
        radio = f_env(env) / g_scr(screen)
        bright = max([0, min([ulimit, stride * round(steps * radio)])])
        print('\t\t'.join([f'\rEnv: {env:.3f}', f'Screen: {screen:.3f} / {mean:.3f}',
                           f'Bright: {bright}']), end='')
        # 设置屏幕亮度
        if bright != cur_br:
            set_brightness(bright)
            cur_br = bright
        # 更新画面亮度的滑动平均值, 判断是否休眠
        mean = (1 - momentum) * mean + momentum * screen
        is_sleep = abs(screen - mean) < 1e-3
        is_active = abs(screen - mean) > 6e-3
        momentum = .2 if is_sleep or is_active else 1e-3
        time.sleep(interval if is_sleep else 0.1)


if __name__ == '__main__':
    # Monitor()
    while True:
        try:
            SightSaver()
        except:
            print('\nAn exception has occurred, waiting to retry')
            time.sleep(60)
