import logging
import os
import pathlib
from functools import wraps

from tqdm import tqdm

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


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


class call_counter:
    """ Function call counter"""

    def __init__(self, func):
        self.__func = func
        self.__cnt = 0

    def __call__(self, *args, **kwargs):
        self.__cnt += 1
        return self.__func(*args, **kwargs)

    def __int__(self):
        return self.__cnt

    def __repr__(self):
        return f"<function {self.__func.__name__} : call_cnt = {self.__cnt}>"


def try_except(func):
    # try-except function. Usage: @try_except decorator
    @wraps(func)
    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            LOGGER.error(error)

    return handler


def parse_txt_cfg(file, encoding="utf-8", comments="#"):
    """
    Supported syntax:
        content     # comment
    """
    with open(file, encoding=encoding) as f:
        for i_line in enumerate(s.split(comments)[0].strip() for s in f.read().splitlines()):
            if i_line[1]: yield i_line


class Path(pathlib.WindowsPath if os.name == "nt" else pathlib.PosixPath, pathlib.Path):

    def fsize(self, unit: str = "B"):
        size = self.stat().st_size if self.is_file() else (
            sum(p.stat().st_size for p in self.glob("**/*") if p.is_file()))
        return size / 1024 ** ("B", "KB", "MB", "GB").index(unit)

    def lazy_obj(self, fget, **fld_kwd):
        f_load_dump = {
            "json": self.json, "yaml": self.yaml, "csv": self.csv, "xlsx": self.excel, "pt": self.torch
        }.get(self.suffix[1:], self.binary)
        # load the data using the load/dump method
        if self.is_file():
            data = f_load_dump(None, **fld_kwd)
            LOGGER.info(f"Load <{type(data).__name__}> from {self}")
        else:
            self.parent.mkdir(parents=True, exist_ok=True)
            data = fget()
            f_load_dump(data, **fld_kwd)
        return data

    def collect_file(self, formats: str | list[str]):
        formats = [formats] if isinstance(formats, str) else formats
        pools = [], []
        # Collect all files in this directory
        qbar = tqdm(self.rglob("*.*"))
        for f in qbar:
            qbar.set_description(f"Collecting files")
            pools[f.suffix[1:] in formats].append(f)
        # Warning if any file does not conform to the format
        if pools[False]:
            LOGGER.warning(f"Unsupported files: {', '.join(map(str, pools[False]))}")
        return pools[True]

    def copy_to(self, dst):
        import shutil
        shutil.copy(self.resolve(), dst.resolve())

    def binary(self, data=None, **kwargs):
        import pickle
        return pickle.loads(self.read_bytes(), **kwargs) \
            if data is None else self.write_bytes(pickle.dumps(data, **kwargs))

    def json(self, data=None, **kwargs):
        import json
        return json.loads(self.read_text(), **kwargs) \
            if data is None else self.write_text(json.dumps(data, indent=4, **kwargs))

    def yaml(self, data=None, **kwargs):
        import yaml
        return yaml.load(self.read_text(), Loader=yaml.Loader, **kwargs) \
            if data is None else self.write_text(yaml.dump(data, **kwargs))

    def csv(self, data=None, **kwargs):
        import pandas as pd
        return pd.read_csv(self, **kwargs) \
            if data is None else data.to_csv(self, **kwargs)

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
        f.extractall(path or self.parent, pwd=pwd)


def concat_txt(src: list[Path],
               dst: Path,
               encoding: str = "utf-8"):
    with dst.open("w", encoding=encoding) as fo:
        for p in src:
            fo.write(p.read_text(encoding=encoding) + "\n")


if __name__ == "__main__":
    os.chdir(r"D:\Information\计算方法\hw")
    concat_txt(sorted(Path().glob("hw*.md"), key=lambda x: int(x.stem[2:])),
               Path("concat.md"))
