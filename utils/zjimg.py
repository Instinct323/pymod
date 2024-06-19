import logging
import random
import string
from itertools import product
from typing import Callable

from tqdm import tqdm

from ..zjdl.utils.imgtf import *

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def star_trails(src: Iterable[np.ndarray],
                decay: float = 0.99,
                agg_fun: Callable = np.maximum):
    """ 制作星轨
        :param src: 图像数组 [B, H, W, C]
        :param decay: 亮度衰减系数"""
    assert 0 < decay <= 1
    src = iter(tqdm(src))
    # 处理第一张图像
    cur = next(src)
    yield cur
    cur = cur.astype(np.float32)
    # 处理后续图像
    for img in src:
        cur = agg_fun(cur * decay, img.astype(np.float32))
        yield np.round(cur).astype(np.uint8)


class AsciiArt:

    def __init__(self, char="#$@&GU?^.    "):
        self.lut = np.linspace(0, len(char), 256, dtype=np.uint8)
        self.char = np.array(list(char + " "))

    def __call__(self, img, rows=20):
        img = cv2.imread(str(img), flags=cv2.IMREAD_GRAYSCALE)
        if isinstance(img, np.ndarray):
            scale = rows / img.shape[0] * np.array([1, 2.8])
            # 得到新的尺寸
            new_shape = np.round(np.array(img.shape) * scale).astype(np.int32)
            img = cv2.resize(img, new_shape[::-1])
            stream = self.char[cv2.LUT(img, self.lut)]
            # 拼接字符串
            stream = map(lambda seq: "".join(seq).rstrip(), stream)
            return "\n".join(stream)


class DictOrder:

    def __init__(self,
                 length: int = 6,
                 skip: int = random.randint(4, 14),
                 char: str = string.ascii_lowercase):
        self.skip = skip
        self.iter = product(char, repeat=length)

    def __iter__(self):
        def generator():
            for i, char in enumerate(self.iter):
                if i % self.skip == 0: yield "".join(char)

        return generator()


class Puzzle:
    """ :param img: 前景图像文件
        :param material: 隐藏图像素材包的路径
        :param shape: 前景图像的目标分割形状
        :param dpi: 前景图像的宽度
        :param pad_width: 隐藏图像的侧边距"""

    def __init__(self,
                 img: Path,
                 material: Path,
                 shape: tuple = (2, 3),
                 dpi: int = 1280,
                 pad_width: float = 0.05,
                 pad_value: int = 255):
        # 对前景图像进行分割, 并读取隐藏图像的素材包
        self.stride = dpi
        self.cells = self.partition(img, shape)
        self.pad_kwarg = dict(borderType=cv2.BORDER_CONSTANT,
                              value=[pad_value] * 3 if isinstance(pad_value, int) else pad_value)
        # 计算边界填充参数: w, h
        self.pad_size = np.round(np.array([pad_width, pad_width / 2]) / 2 * self.stride).astype(np.int32)
        self.material = self.parse_material(material)
        self.puzzle()

    def imread(self, file, warn_only=True):
        img = cv2.imread(str(file))
        check = isinstance(img, np.ndarray)
        # 无法读取时发出警告
        if not check:
            msg = f"Unreadable image file: {file}"
            if warn_only:
                LOGGER.warning(msg)
            else:
                raise RuntimeError(msg)
        else:
            ksize = int(img.shape[1] / self.stride / 2) * 2 + 1
            if ksize >= 3: img = cv2.GaussianBlur(img, ksize=[ksize] * 2, sigmaX=1)
        return img, check

    def partition(self, img, shape):
        """ 前景图像分割"""
        img = self.imread(img, warn_only=False)[0]
        # self.stride = max(round(img.shape[i] / shape[i]) for i in range(2))
        img = cv2.resize(img, np.array(shape[::-1]) * self.stride)
        # 对图像分割到若干个单元格
        cells = sum(map(lambda x: np.split(x, shape[1], axis=1),
                        np.split(img, shape[0], axis=0)), [])
        return cells

    def parse_material(self, material):
        """ 隐藏图像素材包解析"""
        if material:
            material = [folder for folder in material.iterdir() if folder.is_dir()]
            assert len(material) == len(self.cells), f"The number of material packs should be {len(self.cells)}"
            # 素材图像的宽度
            w = self.stride - 2 * self.pad_size[0]
            for i in np.arange(len(material)):
                # 取出素材包路径
                orders = iter(DictOrder())
                folder, material[i] = material[i], []
                for file in folder.iterdir():
                    j = Path()
                    while j.exists():
                        j = file.parent / f"{next(orders)}{file.suffix}"
                    file = file.rename(j)
                    img, check = self.imread(file)
                    # 进行边界填充并进行尺寸变换
                    if check:
                        # 素材图像的高度
                        h = round(img.shape[0] / img.shape[1] * w)
                        img = cv2.resize(img, [w, h])
                        img = cv2.copyMakeBorder(img, *self.pad_size.repeat(2)[::-1], **self.pad_kwarg)
                        material[i].append(img)
                LOGGER.info(f"The material package {i + 1} is loaded")
        return material

    def puzzle(self):
        """ 拼接前景图像与素材包图像"""
        concat = lambda x: np.concatenate(x, axis=0) if x else np.array([])
        pad_vert = lambda x, bottom, top: (
            cv2.copyMakeBorder(x, bottom=bottom, top=top, left=0, right=0, **self.pad_kwarg)) \
            if x.size else np.full((bottom + top, self.stride, 3), 255, dtype=np.uint8)

        for i, cell in enumerate(self.cells):
            img_queue = self.material[i]
            # np.random.shuffle(img_queue)
            if img_queue:
                # 不考虑前景图像的情况下, 沿 y 轴拼接之后素材图像底边的位置
                img_h = np.array([img.shape[0] for img in img_queue], dtype=np.float32)
                loc_bottom = np.cumsum(img_h)
                # 找到底边最靠近 y 轴中点的图像, +1 得到前景图像插入位置
                length = len(img_queue)
                loss = np.abs(loc_bottom / loc_bottom[-1] - 0.5) * \
                       (1 + np.abs(np.arange(length) + 0.5 - length / 2))
                ctr = loss.argmin() + 1
                # 分别对两队列中的图像进行拼接, 并填充边界
                img_queue = img_queue[:ctr], img_queue[ctr:]
                img_queue = list(map(concat, img_queue))
                max_h = max(j.shape[0] for j in img_queue)
                img_queue[0] = pad_vert(img_queue[0], bottom=0, top=max_h - img_queue[0].shape[0] + self.pad_size[1])
                img_queue[1] = pad_vert(img_queue[1], bottom=max_h - img_queue[1].shape[0] + self.pad_size[1], top=0)
                # 填充前景图像的边界
                cell = pad_vert(cell, bottom=self.pad_size[1] * 4, top=self.pad_size[1] * 4)
                img_queue.insert(1, cell)
                cell = np.concatenate(img_queue)
            cv2.imwrite(f"{i + 1}_puzzle.png", cell)
            LOGGER.info(f"The part {i + 1} has been saved as {i + 1}_puzzle.png")
        LOGGER.info(f"The generated image has been saved in {Path.cwd()}")


if __name__ == "__main__":
    import os

    # exp 1: n 宫格长图
    os.chdir(Path("D:/Workbench/data/Camera"))
    Puzzle(Path("exp.jpg"), material=Path(), shape=(3, 3), dpi=2000, pad_value=255)

    # exp 2: 星轨制作
    # 生成序列
    src = Path("D:/Information/Data/dataset/dali-star-trails")
    raw = src.parent / "raw.mp4"
    if src.is_dir() and not raw.is_file():
        img2video(src.iterdir(), raw)
    # 星轨效果
    target = src.parent / "target.mp4"
    dst = src.parent / "final.mp4"
    if target.is_file() and not dst.is_file():
        result = []
        for img in star_trails(VideoCap(target)):
            result.append(img)
            cv2.imshow("s", img)
            cv2.waitKey(1)
        img2video(result, dst)
