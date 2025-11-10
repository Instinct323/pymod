import re
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

BG_COLOR = 114
# {"bmp", "webp", "pfm", "ppm", "ras", "pnm", "dib", "tiff", "pbm", "pic",
# "hdr", "tif", "sr", "jp2", "jpg", "pgm", "pxm", "exr", "png", "jpe", "jpeg"}
IMG_FORMAT = set(re.findall(r"\\\*\.(\w+)", cv2.imread.__doc__))

to_2tuple = lambda x: x if x is None or isinstance(x, (list, tuple)) else (x,) * 2
clip_abs = lambda x, a: np.clip(x, a_min=-a, a_max=a)


def to_tensor(img: np.ndarray,
              pdim: tuple[int] = (-1, -3, -2)):
    import torch
    img = torch.from_numpy(np.ascontiguousarray(img[..., ::-1]))
    return img.permute(0, *pdim) if img.dim() == 4 else img.permute(*pdim)


def load_img(file: str | Path,
             img_size: int = None) -> np.ndarray:
    bgr = cv2.imread(str(file))
    assert isinstance(bgr, np.ndarray), f"Error loading data from {file}"
    if img_size:
        bgr = sv.resize_image(bgr, [img_size] * 2, keep_aspect_ratio=True)
    return bgr


def fsize_lim_save(file: Path,
                   img: np.ndarray,
                   fsize: int = 2 ** 20,
                   eps: float = 1e-3,
                   max_iter: int = 100):
    """ 限制图像文件大小"""
    cv2.imwrite(str(file), img)
    org = file.stat().st_size
    assert fsize < org, f"File size is already less than the target size: {org}"
    capa_tar = 1 - eps / 2
    # 反馈调节
    r_img = fsize * capa_tar / org
    capacity = [1]
    best = (0, None)
    for i in range(max_iter):
        # 图像缩放, 最大缩放比例为 2
        tmp = sv.scale_image(img, min(2, r_img * capa_tar / capacity[-1]))
        r_img = tmp.shape[0] / img.shape[0]
        cv2.imwrite(str(file), tmp)
        capacity.append(file.stat().st_size / fsize)
        # 保存最好的结果
        print(f"[INFO] Iteration {i}: \tcapacity = {capacity[-1]:.6f}, r_img = {r_img:.6f}")
        if capacity[-1] < 1:
            if capacity[-1] > best[0]: best = (capacity[-1], r_img)
            if capacity[-1] > 1 - eps: break
        # 循环检测
        if capacity[-1] in capacity[-3:-1]:
            capa_tar = 1 - eps * np.random.random()
    # 重新缩放图像
    cv2.imwrite(str(file), sv.scale_image(img, best[1])[0])
    return best[1]


def check_imgfile(file: Path):
    """ 检查图像文件状态"""
    assert file.is_file(), f"File not found: {file}"
    assert file.suffix[1:] in IMG_FORMAT, f"Unsupported image format: {file.suffix}"
    assert not re.search(r"[\u4e00-\u9fa5]", file.stem), f"Invalid image name: {file}"
    return file


class VideoSink(sv.VideoSink):

    def __init__(self,
                 dst: str | Path,
                 width: int = 1920,
                 aspect_radio: float = 4 / 3,
                 fps: int = 30,
                 pad: int = 255):
        """
        :param dst: 视频文件名称 (*.mp4)
        :param width: 视频宽度
        :param aspect_radio: 视频宽高比
        :param fps: 视频帧率
        """
        super().__init__(str(dst), sv.VideoInfo(width=width, height=round(width / aspect_radio), fps=fps))
        self.pad = [pad] * 3

    def write_frame(self, img: str | Path | np.ndarray):
        if not isinstance(img, np.ndarray):
            # 从其他数据类型加载图像
            if isinstance(img, (str, Path)):
                img = load_img(img)
            else:
                raise TypeError("Unrecognized image type")
        super().write_frame(sv.letterbox_image(img, self.video_info.resolution_wh, self.pad))


if __name__ == "__main__":
    root = Path(r"C:\Downloads")

    print(fsize_lim_save(root / "scaled.png", load_img(root / "test.png"), 1.5 * 2 ** 20))
