from utils.data import *
from utils.utils import *


class CocoDetect:
    root = Path(r'D:\Information\Python\Library\DataSet\COCO\COCO-detect-20')

    def __new__(cls, aughyp):
        cache_t = (cls.root / 'train.cache').lazy_obj(cls.make_index,
                                                      imgdir=cls.root / 'images/train2017',
                                                      labeldir=cls.root / 'labels/train2017')
        train = MosaicDataset(ImagePool(*cache_t), aughyp=aughyp.yaml())
        cache_v = (cls.root / 'val.cache').lazy_obj(cls.make_index,
                                                    imgdir=cls.root / 'images/val2017',
                                                    labeldir=cls.root / 'labels/val2017')
        val = MosaicDataset(ImagePool(*cache_v))
        return train, val

    @staticmethod
    def make_index(imgdir: Path,
                   labeldir: Path):
        img = imgdir.collect_file(formats=IMG_FORMAT)
        label = []
        for f in tqdm(img, 'Loading labels'):
            f = labeldir / f'{f.stem}.txt'
            v = np.array(list(map(float, f.read_text().split()))).reshape(-1, 5)
            label.append(v)
        return img, label


TRAIN_SET, VAL_SET = CocoDetect(Path('config/mosaic.yaml'))
