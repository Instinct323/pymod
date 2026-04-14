from pathlib import Path

import cv2
import numpy as np
import supervision as sv


def load_img(file: str | Path,
             img_size: int = None) -> np.ndarray:
    bgr = cv2.imread(str(file))
    assert isinstance(bgr, np.ndarray), f"Error loading data from {file}"
    if img_size:
        bgr = sv.resize_image(bgr, [img_size] * 2, keep_aspect_ratio=True)
    return bgr


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
            if isinstance(img, (str, Path)):
                img = load_img(img)
            else:
                raise TypeError("Unrecognized image type")
        super().write_frame(sv.letterbox_image(img, self.video_info.resolution_wh, self.pad))


if __name__ == "__main__":
    pass
