import os
import pathlib
from typing import Callable


class Path(pathlib.WindowsPath if os.name == "nt" else pathlib.PosixPath, pathlib.Path):

    def copy_to(self, dst):
        import shutil
        shutil.copy(self.resolve(), dst.resolve())

    @property
    def f_load_dump(self) -> Callable:
        return {
            "yaml": self.yaml, "yml": self.yaml,  # yaml
            "json": self.json, "csv": self.csv,  # others
        }.get(self.suffix[1:], self.binary)

    def fsize(self, unit: str = "B"):
        size = self.stat().st_size if self.is_file() else (
            sum(p.stat().st_size for p in self.glob("**/*") if p.is_file()))
        return size / 1024 ** ("B", "KB", "MB", "GB").index(unit)

    def lazy_obj(self, fget, **fld_kwd):
        # load the data using the load/dump method
        if self.is_file():
            data = self.f_load_dump(None, **fld_kwd)
        else:
            self.parent.mkdir(parents=True, exist_ok=True)
            data = fget()
            self.f_load_dump(data, **fld_kwd)
        return data

    # functions for load / dump
    def binary(self, data=None, **kwargs):
        import pickle
        return pickle.loads(self.read_bytes(), **kwargs) \
            if data is None else self.write_bytes(pickle.dumps(data, **kwargs))

    def csv(self, data=None, **kwargs):
        import pandas as pd
        return pd.read_csv(self, **kwargs) \
            if data is None else data.to_csv(self, **kwargs)

    def json(self, data=None, **kwargs):
        import json
        return json.loads(self.read_text(), **kwargs) \
            if data is None else self.write_text(json.dumps(data, indent=4, **kwargs))

    def yaml(self, data=None, **kwargs):
        import yaml
        return yaml.load(self.read_text(), Loader=yaml.Loader, **kwargs) \
            if data is None else self.write_text(yaml.dump(data, **kwargs))
