import logging
from functools import wraps

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


if __name__ == "__main__":
    pass
