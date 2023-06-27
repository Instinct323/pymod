import json
import logging
import os
import pickle
import re
import sys
from pathlib import WindowsPath, PosixPath, Path as _path

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# {'bmp', 'webp', 'pfm', 'ppm', 'ras', 'pnm', 'dib', 'tiff', 'pbm', 'pic',
# 'hdr', 'tif', 'sr', 'jp2', 'jpg', 'pgm', 'pxm', 'exr', 'png', 'jpe', 'jpeg'}
IMG_FORMAT = set(re.findall(r'\\\*\.(\w+)', cv2.imread.__doc__))


def collect_files(folder, formats=IMG_FORMAT):
    formats = [formats] if isinstance(formats, str) else formats
    pools = [], []
    # 收集该目录下的所有文件
    qbar = tqdm(folder.glob('**/*.*'))
    for f in qbar:
        qbar.set_description(f'Collecting files')
        pools[f.suffix[1:] in formats].append(f)
    # 如果有文件不符合格式, 则警告
    if pools[False]:
        LOGGER.warning(f'Unsupported files: {", ".join(map(str, pools[False]))}')
    return pools[True]


def colorstr(*args):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = args if len(args) > 1 else ('blue', 'bold', args[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


class timer:

    def __init__(self, repeat: int = 1, avg: bool = True):
        self.repeat = max(1, int(repeat) if isinstance(repeat, float) else repeat)
        self.avg = avg

    def __call__(self, func):
        def handler(*args, **kwargs):
            import time
            start = time.time()
            result = tuple(func(*args, **kwargs) for i in range(self.repeat) if not i)[0]
            cost = (time.time() - start) * 1e3
            print(f'{func.__name__}: {cost / self.repeat if self.avg else cost:.3f} ms')
            return result

        return handler


class run_once:

    def __init__(self, interval: float = 20.):
        self.interval = interval

    def __call__(self, func):
        def handler(*args, **kwargs):
            import time
            # Try to run it an infinite number of times until it succeeds
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    LOGGER.warning(error)
                    time.sleep(self.interval)

        return handler


def singleton(cls):
    # Singleton class decorators
    _instance = {}

    def handler(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return handler


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            LOGGER.warning(error)

    return handler


def select_file(handler, glob_pats=('*.pdf',), wo_app=True):
    from PyQt5.QtWidgets import QFileDialog, QApplication
    # 请执行 sys.exit(0) 退出程序
    if wo_app: app = QApplication(sys.argv)
    dialog = QFileDialog()
    file = dialog.getOpenFileName(caption='Select file',
                                  filter=('; '.join(glob_pats).join('()') if glob_pats else None))[0]
    if file: return handler(Path(file))


def select_dir(handler, wo_app=True):
    from PyQt5.QtWidgets import QFileDialog, QApplication
    # 请执行 sys.exit(0) 退出程序
    if wo_app: app = QApplication(sys.argv)
    dialog = QFileDialog()
    dire = dialog.getExistingDirectory(None, 'Select directory')
    if dire: return handler(Path(dire))


class Path(WindowsPath if os.name == 'nt' else PosixPath, _path):

    def fsize(self, unit: str = 'B'):
        return self.stat().st_size / 1024 ** ('B', 'KB', 'MB', 'GB').index(unit) if self.is_file() else 0.

    def lazy_obj(self, fget, *args, **kwargs):
        f_load_dump = {
            'json': self.json, 'yaml': self.yaml,
            'csv': self.csv, 'xls': self.excel
        }.get(self.suffix[1:], self.pickle)
        # 根据 load/dump 方法载入数据
        if self.is_file():
            data = f_load_dump()
            LOGGER.info(f'Load <{type(data).__name__}> from {self}')
        else:
            data = fget(*args, **kwargs)
            f_load_dump(data)
        return data

    def pickle(self, data=None, **kwargs):
        f = self.open('rb') if data is None else self.open('wb')
        return pickle.load(f, **kwargs) if data is None else pickle.dump(data, f, **kwargs)

    def json(self, data=None, **kwargs):
        f = self.open('r') if data is None else self.open('w')
        return json.load(f, **kwargs) if data is None else json.dump(data, f, indent=4, **kwargs)

    def yaml(self, data=None, **kwargs):
        return yaml.load(self.read_text(), Loader=yaml.Loader, **kwargs) \
            if data is None else self.write_text(yaml.dump(data, **kwargs))

    @try_except
    def csv(self, data: pd.DataFrame = None, **kwargs):
        return pd.read_csv(self, **kwargs) if data is None else data.to_csv(self, **kwargs)

    @try_except
    def excel(self, data: pd.DataFrame = None, **kwargs):
        # Only excel in 'xls' format is supported
        if data is None: return pd.read_excel(self, **kwargs)
        writer = pd.ExcelWriter(self)
        for df in [data] if isinstance(data, pd.DataFrame) else data:
            df.to_excel(writer, **kwargs)
        writer.save()


class Capture(cv2.VideoCapture):
    ''' 视频捕获
        file: 视频文件名称 (默认连接摄像头)
        show: 视频帧的滞留时间 (ms)
        dpi: 相机分辨率'''

    def __init__(self,
                 file: str = 0,
                 show: int = 0,
                 dpi: list = [1280, 720]):
        super(Capture, self).__init__(file)
        if not self.isOpened():
            raise RuntimeError('Failed to initialize video capture')
        self.show = show
        # 设置相机的分辨率
        if not file and dpi:
            self.set(cv2.CAP_PROP_FRAME_WIDTH, dpi[0])
            self.set(cv2.CAP_PROP_FRAME_HEIGHT, dpi[1])

    def __iter__(self):
        return self

    def __next__(self):
        ok, image = self.read()
        if not ok: raise StopIteration
        if self.show:
            cv2.imshow('Capture', image)
            cv2.waitKey(self.show)
        return image


if __name__ == '__main__':
    p = Path(r'D:\Information\Download\CCF.xls')
    print(p.pickle.__name__)
