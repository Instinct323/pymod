from typing import Callable

from tqdm import tqdm

from pymod.utils.zjcv import *


def star_trails_video(
        src: Iterable[np.ndarray],
        decay: float = 0.99,
        agg_fun: Callable = np.maximum):
    """ 制作星轨视频
        :param src: 图像数组 [B, H, W, C]
        :param decay: 亮度衰减系数"""
    assert 0 < decay <= 1
    src = iter(tqdm(src))
    # 处理第一张图像
    cur = next(src)
    yield cur
    cur = cur.astype(np.float32)
    # 处理后续图像
    for img in map(np.float32, src):
        cur = agg_fun(cur * decay, img)
        yield np.round(cur).astype(np.uint8)


def star_trails_image(
        src: Iterable[np.ndarray],
        weight: Iterable[float],
        agg_fun: Callable = np.maximum):
    """ 制作星轨图像
        :param src: 图像数组 [B, H, W, C]
        :param weight: 图像权重"""
    cur = None
    for w, img in zip(weight, map(np.float32, src)):
        assert 0 <= w <= 1
        cur = w * img if cur is None else agg_fun(cur, w * img)
    return np.round(cur).astype(np.uint8)


if __name__ == "__main__":
    i = 0

    # exp 1: 星轨视频制作
    if i == 0:
        # 生成序列
        src = Path("D:/Information/Data/dataset/dali-star-trails")
        raw = src.parent / "raw.mp4"
        if src.is_dir() and not raw.is_file():
            with VideoWriter(raw) as vw:
                for img in src.iterdir():
                    vw.write(img)
        # 星轨效果
        target = src.parent / "target.mp4"
        dst = src.parent / "final.mp4"
        if target.is_file() and not dst.is_file():
            with VideoWriter(dst) as vw:
                for img in star_trails_video(VideoCap(target)):
                    vw.write(img)
                    cv2.imshow("s", img)
                    cv2.waitKey(1)

    # exp 2: 星轨图像制作
    elif i == 1:
        src = Path("D:/Information/Data/dataset/moliugong-star-trails-2")
        files = list(src.iterdir())
        files.reverse()
        img = star_trails_image(map(cv2.imread, map(str, files)), np.linspace(0.5, 1, len(files)))
        cv2.imwrite(str(src.parent / "final.jpg"), img)
