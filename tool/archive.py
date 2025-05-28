import math
import shutil
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

from pymod.utils.utils import LOGGER, parse_txt_cfg

to_2tuple = lambda x: x if x is None or isinstance(x, (list, tuple)) else (x,) * 2


def make_blurred_border(src: np.ndarray,
                        img_size: Union[int, tuple[int, int]] = None,
                        aspect_ratio: float = None):
    """
    生成带有模糊边框的图像
    :param src: 原图像 (OpenCV 格式)
    :param img_size: 目标尺寸
    :param aspect_ratio: 长:宽
    """
    int_round = lambda x: np.round(x).astype(np.int64)
    src_size = np.array(src.shape[1::-1])
    img_size = np.array(to_2tuple(img_size))
    if not img_size and aspect_ratio:
        # 根据长宽比计算目标尺寸, 取面积最大者
        s1 = src_size[1] / np.array((aspect_ratio, 1))
        s2 = src_size[0] * np.array((1, aspect_ratio))
        img_size = int_round(max((s1, s2), key=np.prod))
    assert np.all(img_size), "No target size specified."
    rs = np.sort(img_size / src_size)
    # 前景图像
    fg_size = int_round(rs[0] * src_size)
    fg = cv2.resize(src, fg_size)
    # 背景图像
    bg_size = int_round(rs[1] * src_size)
    bg = cv2.resize(src, bg_size)
    pad_size = img_size - fg_size
    if np.any(pad_size >= 2) and np.abs(1 - max(pad_size / img_size)) > 0.02:
        # 裁剪背景, 分割出填充区域
        overflow = (bg_size - img_size) // 2
        bg = bg[overflow[1]:, overflow[0]:][:img_size[1], :img_size[0]]
        axis = patches = None
        if pad_size[0] > 0:
            axis = 1
            patches = bg[:, :pad_size[0] // 2], bg[:, pad_size[0] // 2 - pad_size[0]:]
        elif pad_size[1] > 0:
            axis = 0
            patches = bg[:pad_size[1] // 2], bg[pad_size[1] // 2 - pad_size[1]:]
        # 对背景进行滤波
        if patches:
            r = max(20 / max(pad_size), 200 / min(img_size))
            patches = tuple(map(
                lambda x: cv2.resize(
                    cv2.GaussianBlur(
                        cv2.resize(x, None, None, r, r),
                        (11,) * 2, 0),
                    x.shape[1::-1]),
                patches))
        # 合并前景和背景
        fg = np.concatenate((patches[0], fg, patches[1]), axis=axis)
    elif np.any(pad_size):
        fg = cv2.resize(fg, img_size)
    return fg


class FileArchiver:
    """
    文件有序归档管理器
    :param file_fmt: 文件名格式, %i 序号, %n 文件名
    """

    def __init__(self,
                 txt_cfg: Union[str, Path],
                 dst: Union[str, Path],
                 file_fmt: str = "%i-%n",
                 reverse: bool = False):

        txt_cfg = Path(txt_cfg)
        assert txt_cfg.is_file(), "Invalid configuration file."
        # 初始化归档目录
        self.dst = Path(dst) / f"{__class__.__name__}-{time.strftime('%Y%m%d%H%M%S')}"
        self.dst.mkdir(parents=True)
        # 读取配置文件
        self.files: list[Path] = []
        self.include(txt_cfg)
        if reverse: self.files.reverse()
        # 执行归档
        ndigit = math.ceil(math.log10(len(self.files)))
        with (self.dst / ".archive.txt").open("w") as fi:
            for i, f in enumerate(tqdm(self.files, desc="Archiving")):
                fi.write(str(f) + "\n")
                name = file_fmt.replace("%n", f.stem).replace("%i", str(i).zfill(ndigit))
                shutil.copy(f, self.dst / (name + f.suffix))

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
    fa = FileArchiver(r"D:\Information\Source\image-HQ\archive.txt", dst=r"D:\Downloads", file_fmt="%i")
    quit()
    for f in tqdm(fa.dst.iterdir(), total=len(fa.files), desc="Making blurred border"):
        if f.is_file() and f.suffix != ".txt":
            img = make_blurred_border(cv2.imread(str(f)), aspect_ratio=3 / 4)
            cv2.imwrite(str(f), img)
