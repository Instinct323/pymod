import re
from collections import Counter
from pathlib import Path
from typing import Union, Iterable

import cv2
import numpy as np

BG_COLOR = 114
# {"bmp", "webp", "pfm", "ppm", "ras", "pnm", "dib", "tiff", "pbm", "pic",
# "hdr", "tif", "sr", "jp2", "jpg", "pgm", "pxm", "exr", "png", "jpe", "jpeg"}
IMG_FORMAT = set(re.findall(r"\\\*\.(\w+)", cv2.imread.__doc__))

to_2tuple = lambda x: x if x is None or isinstance(x, (list, tuple)) else (x,) * 2
clip_abs = lambda x, a: np.clip(x, a_min=-a, a_max=a)


def to_tensor(img, pdim=(-1, -3, -2)):
    import torch
    img = torch.from_numpy(np.ascontiguousarray(img[..., ::-1]))
    return img.permute(0, *pdim) if img.dim() == 4 else img.permute(*pdim)


def resize(bgr, img_size):
    h, w = bgr.shape[:2]
    img_size = to_2tuple(img_size)
    r = min(img_size[1] / h, img_size[0] / w)
    new_shape = tuple(map(round, (h * r, w * r)))
    if new_shape != (h, w):
        bgr = cv2.resize(bgr, new_shape[::-1])
    return bgr, r


def load_img(file: Union[str, Path],
             img_size: int = None) -> np.ndarray:
    bgr = cv2.imread(str(file))
    assert isinstance(bgr, np.ndarray), f"Error loading data from {file}"
    if img_size:
        bgr = resize(bgr, img_size)[0]
    return bgr


def check_imgfile(file: Path):
    """ 检查图像文件状态"""
    assert file.is_file(), f"File not found: {file}"
    assert file.suffix[1:] in IMG_FORMAT, f"Unsupported image format: {file.suffix}"
    assert not re.search(r"[\u4e00-\u9fa5]", file.stem), f"Invalid image name: {file}"
    return file


def letter_box(bgr, img_size=(640, 640), pad=BG_COLOR, stride=None):
    """ 边界填充至指定尺寸"""
    img_size = to_2tuple(img_size)
    pad = (pad,) * 3 if isinstance(pad, int) else pad
    bgr, r = resize(bgr, img_size)
    # 放缩后的原始尺寸
    h, w = bgr.shape[:2]
    dh, dw = img_size[1] - h, img_size[0] - w
    # 最小化边界尺寸
    if stride: dh, dw = map(lambda x: x % stride, (dh, dw))
    dh, dw = map(lambda x: x / 2, (dh, dw))
    # 添加边界
    top, bottom = map(round, (dh - 0.1, dh + 0.1))
    left, right = map(round, (dw - 0.1, dw + 0.1))
    bgr = cv2.copyMakeBorder(bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad)  # add border
    return bgr, r, (dh, dw)


def img_mul(img, alpha):
    img = img.astype(np.float16)
    return np.uint8(np.clip((img * alpha).round(), a_min=0, a_max=255))


def img2video(src: Iterable,
              dst: Union[Path, str],
              width: int = 1920,
              aspect_radio: float = 4 / 3,
              fps: int = 30,
              pad: int = 255):
    """ 图像序列转视频
        :param src: 图像文件列表 / 图像数组
        :param dst: 视频文件名称
        :param width: 视频宽度
        :param aspect_radio: 视频宽高比
        :param fps: 视频帧率
        :param pad: 边界填充颜色"""
    img_size = width, round(width / aspect_radio)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    assert Path(dst).suffix == ".mp4", "The video format must be mp4"
    # 逐帧写入视频
    video = cv2.VideoWriter(str(dst), fourcc, fps, img_size)
    for img in src:
        if not isinstance(img, np.ndarray):
            # 从其他数据类型加载图像
            if isinstance(img, (str, Path)):
                img = load_img(img)
            else:
                raise TypeError("Unrecognized image type")
        video.write(letter_box(img, img_size, pad=pad)[0])
    video.release()


class VideoCap(cv2.VideoCapture):
    """ 视频捕获
        :param src: 视频文件名称 (默认连接摄像头)
        :param delay: 视频帧的滞留时间 (ms)
        :param dpi: 相机分辨率"""

    def __init__(self,
                 src: Union[int, str, Path] = 0,
                 delay: int = 0,
                 dpi: list = None):
        src = str(src) if isinstance(src, Path) else src
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


class _augment:

    def get_param(self) -> dict:
        return {}

    @staticmethod
    def apply(img) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, img):
        return self.apply(img, **self.get_param())


class RandomFlip(_augment):

    def __init__(self, hyp):
        self.flips = hyp.get("fliplr", 0), hyp.get("flipud", 0)
        for p in self.flips: assert 0 <= p <= 1

    def get_param(self):
        flips = np.random.random(2) < self.flips
        return dict(hflip=flips[0], vflip=flips[1])

    @staticmethod
    def apply(img, hflip, vflip, **kwargs):
        if any((hflip, vflip)):
            img = cv2.flip(img, flipCode=(hflip * 2 + vflip) % 3 - 1)
        return img


class RandomCrop(_augment):

    def __init__(self, hyp):
        self.radio = hyp.get("crop", 1)
        assert 0 <= self.radio <= 1

    def get_param(self):
        r = 1 - np.random.uniform(0, self.radio)
        x, y = np.random.uniform(0, 1 - r, 2)
        return dict(x=x, y=y, r=r)

    @staticmethod
    def apply(img, x, y, r, **kwargs):
        if r != 1 or any((x, y)):
            H, W, C = img.shape
            x1, y1, x2, y2 = map(round, (x * W, y * H, (x + r) * W, (y + r) * H))
            img = cv2.resize(img[y1: y2, x1: x2], (W, H))
        return img


class ColorJitter(_augment):

    def __init__(self, hyp):
        self.hue, self.sat, self.value = (hyp.get(f"hsv_{i}", 0) for i in "hsv")
        assert 0 <= self.hue <= .5 and 0 <= self.sat <= 1 and 0 <= self.value <= 1

    def get_param(self):
        return dict(h=clip_abs(np.random.normal(0, self.hue / 2), 1),
                    s=clip_abs(np.random.normal(0, self.sat / 2), 1),
                    v=clip_abs(np.random.normal(0, self.value / 2), .5))

    @staticmethod
    def apply(img, h, s, v, **kwargs):
        h = round(180 * h)
        # 可行性判断
        flag = bool(h), abs(s) > 2e-3, abs(v) > 2e-3
        if any(flag):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # hsv 空间增强
            if flag[0]: img[..., 0] = cv2.LUT(img[..., 0], np.uint8((np.arange(h, h + 256)) % 180))
            if flag[1]: img[..., 1] = cv2.LUT(img[..., 1], img_mul(np.arange(256), s + 1))
            if flag[2]: img[..., 2] = cv2.LUT(img[..., 2], img_mul(np.arange(256), v + 1))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img


class GaussianBlur(_augment):
    """ :param sigma: [min, std]"""

    def __init__(self, hyp, sigma=(0.1, 1.0)):
        self.ksize = hyp.get("gb_kernel", 0)
        self.sigma = sigma

    def get_param(self):
        sigma = abs(np.random.normal(0, self.sigma[1])) + self.sigma[0]
        return dict(k=self.ksize, sigma=sigma)

    @staticmethod
    def apply(img, k, sigma, **kwargs):
        if k: img = cv2.GaussianBlur(img, ksize=(k,) * 2, sigmaX=sigma)
        return img


class Transform(list, _augment):

    def __init__(self, *tfs):
        key, t = sum((Counter(tf.get_param().keys()) for tf in tfs), Counter()).most_common(1)[0]
        assert t == 1, f"Duplicate keyword argument <{key}>"
        super().__init__(tfs)

    def get_param(self):
        param = {}
        for tf in self: param.update(tf.get_param())
        return param

    def apply(self, img, param):
        for tf in self: img = tf.apply(img, **param)
        return img

    def __call__(self, img):
        for tf in self: img = tf(img)
        return img


if __name__ == "__main__":
    # cj = Transform(Path("../cfg/hyp.yaml"))

    img = load_img("../data/dog_cat.jfif", 400)
    img = cv2.warpAffine(img, np.array([[1, 0, 100.],
                                        [0, 1, 100]]), (400,) * 2)

    cv2.imshow("s", img)
    cv2.waitKey(0)
