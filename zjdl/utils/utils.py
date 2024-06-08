import logging
import os
import re
from functools import wraps
from pathlib import WindowsPath, PosixPath, Path as _path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)

STOP_WORDS = r"\/:*?'<>|"


def colorstr(msg, *setting):
    setting = ("blue", "bold") if not setting else ((setting,) if isinstance(setting, str) else setting)
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code
    colors = {
        # basic colors
        "black": "\033[30m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
        "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m", "white": "\033[37m",
        # bright colors
        "bright_black": "\033[90m", "bright_red": "\033[91m", "bright_green": "\033[92m",
        "bright_yellow": "\033[93m", "bright_blue": "\033[94m", "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m", "bright_white": "\033[97m",
        # misc
        "end": "\033[0m", "bold": "\033[1m", "underline": "\033[4m",
    }
    return "".join(colors[x] for x in setting) + str(msg) + colors["end"]


class timer:

    def __init__(self, repeat: int = 1, avg: bool = True):
        self.repeat = max(1, int(repeat) if isinstance(repeat, float) else repeat)
        self.avg = avg

    def __call__(self, func):
        import time

        @wraps(func)
        def handler(*args, **kwargs):
            t0 = time.time()
            for i in range(self.repeat): res = func(*args, **kwargs)
            cost = (time.time() - t0) * 1e3
            cost = cost / self.repeat if self.avg else cost
            print(f"{func.__name__}: {cost:.3f}")
            return res

        return handler


class run_once:

    def __init__(self, interval: float = 20.):
        self.interval = interval

    def __call__(self, func):
        import time

        @wraps(func)
        def handler(*args, **kwargs):
            # Try to run it an infinite number of times until it succeeds
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    LOGGER.warning(error)
                    time.sleep(self.interval)

        return handler


class singleton:
    # Singleton class decorators
    _instance = {}

    def __new__(ctx, cls):
        @wraps(cls)
        def handler(*args, **kwargs):
            if cls not in ctx._instance:
                ctx._instance[cls] = cls(*args, **kwargs)
            return ctx._instance[cls]

        return handler


def try_except(func):
    # try-except function. Usage: @try_except decorator
    @wraps(func)
    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            LOGGER.error(error)

    return handler


class Path(WindowsPath if os.name == "nt" else PosixPath, _path):

    def fsize(self, unit: str = "B"):
        size = self.stat().st_size if self.is_file() else (
            sum(p.stat().st_size for p in self.glob("**/*") if p.is_file()))
        return size / 1024 ** ("B", "KB", "MB", "GB").index(unit)

    def lazy_obj(self, fget, **fld_kwd):
        f_load_dump = {
            "json": self.json, "yaml": self.yaml,
            "csv": self.csv, "xlsx": self.excel,
            "pt": self.torch
        }.get(self.suffix[1:], self.binary)
        # 根据 load/dump 方法载入数据
        if self.is_file():
            data = f_load_dump(None, **fld_kwd)
            LOGGER.info(f"Load <{type(data).__name__}> from {self}")
        else:
            data = fget()
            f_load_dump(data, **fld_kwd)
        return data

    def collect_file(self, formats):
        formats = [formats] if isinstance(formats, str) else formats
        pools = [], []
        # 收集该目录下的所有文件
        qbar = tqdm(self.glob("**/*.*"))
        for f in qbar:
            qbar.set_description(f"Collecting files")
            pools[f.suffix[1:] in formats].append(f)
        # 如果有文件不符合格式, 则警告
        if pools[False]:
            LOGGER.warning(f"Unsupported files: {', '.join(map(str, pools[False]))}")
        return pools[True]

    def copy_to(self, dst):
        import shutil
        shutil.copy(self, dst)

    def binary(self, data=None, **kwargs):
        import pickle
        return pickle.load(self.open("rb"), **kwargs) \
            if data is None else pickle.dump(data, self.open("wb"), **kwargs)

    def json(self, data=None, **kwargs):
        import json
        return json.load(self.open("r"), **kwargs) \
            if data is None else json.dump(data, self.open("w"), indent=4, **kwargs)

    def yaml(self, data=None, **kwargs):
        import yaml
        return yaml.load(self.read_text(), Loader=yaml.Loader, **kwargs) \
            if data is None else self.write_text(yaml.dump(data, **kwargs))

    @try_except
    def csv(self, data=None, **kwargs):
        import pandas as pd
        return pd.read_csv(self, **kwargs) \
            if data is None else data.to_csv(self, **kwargs)

    @try_except
    def excel(self, data=None, **kwargs):
        import pandas as pd
        # Only excel in "xls" format is supported
        if data is None: return pd.read_excel(self, **kwargs)
        writer = pd.ExcelWriter(self)
        for df in [data] if isinstance(data, pd.DataFrame) else data:
            df.to_excel(writer, **kwargs)
        writer.close()

    def torch(self, data=None, map_location=None):
        import torch
        return torch.load(self, map_location=map_location) \
            if data is None else torch.save(data)

    def netron(self):
        import netron
        if self.suffix == ".pt": LOGGER.warning("pytorch model may not be supported")
        return netron.start(str(self))

    def unzip(self, path=None, pwd=None):
        import zipfile
        f = zipfile.ZipFile(self, mode="r")
        f.extractall(self.parent if path is None else path, pwd=pwd)


class VideoCap(cv2.VideoCapture):
    """ 视频捕获
        :param src: 视频文件名称 (默认连接摄像头)
        :param delay: 视频帧的滞留时间 (ms)
        :param dpi: 相机分辨率"""

    def __init__(self,
                 src: str = 0,
                 delay: int = 0,
                 dpi: list = None):
        super().__init__(src)
        if not self.isOpened():
            raise RuntimeError("Failed to initialize video capture")
        self.delay = delay
        # 设置相机的分辨率
        if dpi:
            assert src == 0, "Only camera can set resolution"
            self.set(cv2.CAP_PROP_FRAME_WIDTH, dpi[0])
            self.set(cv2.CAP_PROP_FRAME_HEIGHT, dpi[1])

    def __iter__(self):
        def generator():
            while True:
                ok, image = self.read()
                if not ok: break
                if self.delay:
                    cv2.imshow("frame", image)
                    cv2.waitKey(self.delay)
                yield image
            # 回到开头
            self.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return generator()

    def __len__(self):
        return round(self.get(cv2.CAP_PROP_FRAME_COUNT))

    def flow(self):
        delay, self.delay = self.delay, 0
        gray1 = cv2.cvtColor(next(self), cv2.COLOR_BGR2GRAY)
        for rgb in self:
            gray2 = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            # 两通道, 分别表示像素在 x,y 方向上的位移值
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            yield rgb, flow
            # 光流图的位移: 笛卡尔 -> 极坐标, hue 表示相角, value 表示幅度
            if delay:
                v, h = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.full_like(rgb, fill_value=255)
                hsv[..., 0] = h * 90 / np.pi
                hsv[..., 2] = cv2.normalize(v, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                cv2.imshow("frame", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
                cv2.waitKey(delay)
            gray1 = gray2
        self.delay = delay


if __name__ == "__main__":
    print("hello" + colorstr("hi", "red", "bold") + "dddd")
