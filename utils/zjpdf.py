from pathlib import Path

import fitz
from PyPDF2 import PdfMerger
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import *
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from tqdm import tqdm


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


def merge_pdf(src: Iterator[Path], dst: Path):
    merger = PdfMerger()
    for f in src: merger.append(f)
    merger.write(str(dst))


def pdf2img(file: Path, suffix='.jpg', root='Project', blowup=15):
    ''' :param file: pdf 文件
        :param suffix: 图像后缀名
        :param root: 保存的源目录
        :param blowup: 图像清晰度'''
    root = file.parent / root
    if not root.is_dir(): root.mkdir()
    # 枚举每一页 pdf
    pdf = fitz.open(file)
    for i, page in tqdm(list(enumerate(pdf)), desc='pdf to image'):
        # 转为图像并保存
        pix = page.get_pixmap(matrix=fitz.Matrix(blowup, blowup))
        pix.save(root / (file.stem + f'_{i + 1}{suffix}'))
    pdf.close()


if __name__ == '__main__':
    ROOT = Path(r'D:\Information\Notes\info')
    pdf2img(ROOT / '童赞嘉.pdf')
