from pathlib import Path

import pandas as pd


class Result(pd.DataFrame):
    __exist__ = []
    orient = "index"
    project = Path()
    file = property(lambda self: self.project / "result.json")

    def __init__(self, project: Path, title: tuple):
        self.project = project
        super().__init__(pd.read_json(self.file, orient=self.orient)) \
            if self.file.is_file() else super().__init__(columns=title)
        # 检查项目是否复用
        if project in self.__exist__:
            raise AssertionError(f"Multiple <{__class__.__name__}> are used in {project}")
        self.__exist__.append(project)

    def record(self, metrics, i: int = None):
        i = len(self) if i is None else i
        self.loc[i] = metrics
        super().__init__(self.convert_dtypes())
        self.to_json(self.file, orient=self.orient, indent=4)


if __name__ == "__main__":
    r = Result(Path("__pycache__"), ("re", "ree", "das"))
    r.record((1, 2, "we"))
    r.record((2, 3, "eweqw"))
    r.export()
    r.plot(fitness="re")
