from collections import Counter

from pymod.utils.zjcv import *


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
