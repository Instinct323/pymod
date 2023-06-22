import json
import pickle
import re
import sys
from pathlib import Path

import fitz
import pandas as pd
import yaml
from PyQt5.QtWidgets import QFileDialog, QApplication
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import *
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

from utils import try_except

STOP_WORDS = r'\/:*?"<>|'


class get_size:
    ''' 获取文件大小'''
    divisor = {key: 1024 ** i for i, key in enumerate(['B', 'KB', 'MB', 'GB'])}

    def __new__(cls, file: Path, unit=None):
        assert file.is_file()
        size = file.stat().st_size
        # 如果提供了单位, 则换算
        if unit in cls.divisor: size /= cls.divisor[unit]
        return round(size, 3)


def pickle_size(obj):
    return len(pickle.dumps(obj))


def reset_img_index(root, remove=None):
    ''' latex 图像源处理'''
    remove = ([remove] if isinstance(remove, int) else remove) if remove else []
    imgs = sorted([(int(re.search(r'^\d+', file.name).group()), file)
                   for file in root.iterdir()], key=lambda item: item[0])
    # 保留的 figure, 以及新的索引
    index = sorted(set([i[0] for i in imgs]) - set(remove))
    pin = 1
    for i, file in imgs:
        while i > index[pin - 1]: pin += 1
        if i == index[pin - 1]:
            if i != pin:
                file.rename(file.with_name(f'{pin}{file.name[len(str(pin)):]}'))
        else:
            file.unlink()


@try_except
def excel_dump(dataframe, file, float_format='%.4f'):
    writer = pd.ExcelWriter(file)
    dataframe = [dataframe] if isinstance(dataframe, pd.DataFrame) else dataframe
    for df in dataframe:
        df.to_excel(writer, float_format=float_format)
    writer.save()


def pdf_load(file: Path):
    # 初始化 pdf 解析设备
    rsrcmgr = PDFResourceManager(caching=False)
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interp = PDFPageInterpreter(rsrcmgr, device)
    # 产出页面的 LTPage 对象
    with file.open('rb') as f:
        for _ in map(interp.process_page, PDFPage.get_pages(f)):
            yield device.get_result()


def pdf_to_img(file: Path, suffix='.jpg', root='Project', blowup=5):
    ''' file: pdf 文件
        suffix: 图像后缀名
        root: 保存的源目录
        blowup: 图像清晰度'''
    root = file.parent / root
    if not root.is_dir(): root.mkdir()
    # 枚举每一页 pdf
    pdf = fitz.open(file)
    for i, page in tqdm(list(enumerate(pdf)), desc='pdf to image'):
        # 转为图像并保存
        pix = page.get_pixmap(matrix=fitz.Matrix(blowup, blowup))
        pix.save(root / (file.stem + f'_{i + 1}{suffix}'))
    pdf.close()


def count_code(path, stop_dir=[], ftype='.py'):
    ''' path: 查找目录
        stop_dir: 停用目录
        return: 文件 / 文件夹下的代码总数'''
    count, path = 0, Path(path)
    for element in path.iterdir():
        # 目录不在停用目录中
        if element.is_dir() and element.name not in stop_dir:
            # 递归查找目录
            count += count_code(element, stop_dir=stop_dir, ftype=ftype)
        # 统计行数
        elif element.is_file() and element.suffix == ftype:
            with open(element, 'r', encoding='utf-8') as obj:
                count += len(list(filter(lambda string: re.search(r'\S', string),
                                         obj.readlines())))
    return count


def file_choose():
    ''' 文件选取函数'''
    dialog = QFileDialog()
    filename = dialog.getOpenFileName(caption='选取文件', filter='(*.xlsx; *.xls)')[0]
    # path = dialog.getExistingDirectory(None, '选取路径')

    if filename:
        file = Path(filename)
        # 文件操作执行区

    '''app = QApplication(sys.argv)
    file_choose()
    sys.exit(0)
    # sys.exit(app.exec_())'''


if __name__ == '__main__':
    app = QApplication(sys.argv)
    file_choose()
    sys.exit(0)
    # sys.exit(app.exec_())
