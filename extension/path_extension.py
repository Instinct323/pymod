import os
import pathlib


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
        else:
            self.parent.mkdir(parents=True, exist_ok=True)
            data = fget()
            f_load_dump(data, **fld_kwd)
        return data

    def binary(self, data=None, **kwargs):
        import pickle
        return pickle.loads(self.read_bytes(), **kwargs) \
            if data is None else self.write_bytes(pickle.dumps(data, **kwargs))

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

    def json(self, data=None, **kwargs):
        import json
        return json.loads(self.read_text(), **kwargs) \
            if data is None else self.write_text(json.dumps(data, indent=4, **kwargs))

    def torch(self, data=None, map_location=None):
        import torch
        return torch.load(self, map_location=map_location) \
            if data is None else torch.save(data)

    def yaml(self, data=None, **kwargs):
        import yaml
        return yaml.load(self.read_text(), Loader=yaml.Loader, **kwargs) \
            if data is None else self.write_text(yaml.dump(data, **kwargs))

    def copy_to(self, dst):
        import shutil
        shutil.copy(self.resolve(), dst.resolve())

    def unzip(self, path=None, pwd=None):
        import zipfile
        f = zipfile.ZipFile(self, mode="r")
        f.extractall(path or self.parent, pwd=pwd)
