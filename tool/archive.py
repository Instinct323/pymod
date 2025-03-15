import math
import shutil
import time
from pathlib import Path
from typing import List
from pymod.utils.utils import LOGGER, parse_txt_cfg
from tqdm import tqdm


class FileArchiver:
    """ 文件有序归档管理器 """

    def __init__(self, txt_cfg, dst):
        txt_cfg = Path(txt_cfg)
        assert txt_cfg.is_file(), "Invalid configuration file."
        # 初始化归档目录
        dst = Path(dst) / f"{__class__.__name__}-{time.strftime('%Y%m%d%H%M%S')}"
        dst.mkdir(parents=True)
        # 读取配置文件
        self.files: List[Path] = []
        self.include(txt_cfg)
        # 执行归档
        ndigit = math.ceil(math.log10(len(self.files)))
        with (dst / ".archive.txt").open("w") as fi:
            for i, f in enumerate(tqdm(self.files, desc="Archiving")):
                fi.write(str(f) + "\n")
                shutil.copy(f, dst / (str(i).rjust(ndigit, "0") + "-" + f.name))

    def include(self, txt_cfg):
        root = Path(txt_cfg.parent).resolve()
        for i, line in parse_txt_cfg(txt_cfg):
            # 处理特殊语法
            if line.startswith("@"):
                k, v = line[1:].split("=")
                # @include=<file_stem>
                if k == "include":
                    self.include(root / (v + ".txt"))
                # @root=<path>
                elif k == "root":
                    root = (txt_cfg.parent / v).resolve()
            # 读取文件名称
            else:
                f = root / line
                self.files.append(f) if f.is_file() else (
                    LOGGER.warning(f"File \"{txt_cfg}\", line {i + 1}: \"{f}\" does not exist."))


if __name__ == '__main__':
    FileArchiver(r"D:\Information\Source\image-HQ\archive.txt", dst=r"D:\Downloads")
