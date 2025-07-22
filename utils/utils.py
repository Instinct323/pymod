import logging
import os
from functools import wraps

from pymod.extension.path_extension import Path

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


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


def parse_txt_cfg(file, encoding="utf-8", comments="#"):
    """
    Supported syntax:
        content     # comment
    """
    with open(file, encoding=encoding) as f:
        for i_line in enumerate(s.split(comments)[0].strip() for s in f.read().splitlines()):
            if i_line[1]: yield i_line


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
