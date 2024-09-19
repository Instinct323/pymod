from pathlib import Path
from typing import Iterator

from tqdm import tqdm


def load_pdf(file: Path):
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage

    # 初始化 pdf 解析设备
    rsrcmgr = PDFResourceManager(caching=False)
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interp = PDFPageInterpreter(rsrcmgr, device)
    # 产出页面的 LTPage 对象
    with file.open("rb") as f:
        for _ in map(interp.process_page, PDFPage.get_pages(f)):
            yield device.get_result()


def merge_pdf(src: Iterator[Path], dst: Path):
    from PyPDF2 import PdfMerger

    merger = PdfMerger()
    for f in src: merger.append(f)
    merger.write(str(dst))


def pdf2img(file: Path, suffix=".png", root="Project", blowup=15):
    """ :param file: pdf 文件
        :param suffix: 图像后缀名
        :param root: 保存的源目录
        :param blowup: 图像清晰度"""
    import fitz

    root = file.parent / root
    if not root.is_dir(): root.mkdir()
    # 枚举每一页 pdf
    pdf = fitz.open(file)
    for i, page in tqdm(list(enumerate(pdf)), desc="pdf to image"):
        # 转为图像并保存
        pix = page.get_pixmap(matrix=fitz.Matrix(blowup, blowup))
        pix.save(root / (file.stem + f"-{i + 1}{suffix}"))
    pdf.close()


if __name__ == "__main__":
    # pdf2img(r"D:\Information\Lib\童赞嘉.pdf")
    ROOT = Path(r"D:\Information\Document\毕业设计")
    merge_pdf(ROOT.iterdir(), ROOT.parent / "merge.pdf")
