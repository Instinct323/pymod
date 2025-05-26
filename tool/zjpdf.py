from pathlib import Path
from typing import Iterator, Callable

from tqdm import tqdm


def load_pdf(src: Path,
             pages_filter: Callable[[int], bool] = None):
    """
    :param src: pdf 文件
    :param pages_filter: 过滤函数
    """
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
    with src.open("rb") as f:
        for i, page in enumerate(PDFPage.get_pages(f)):
            # 跳过不需要的页面
            if pages_filter and not pages_filter(i): continue
            interp.process_page(page)
            yield device.get_result()


def pdf2text(src: Path,
             dst: Path,
             pages_filter: Callable[[int], bool] = None):
    """ 提取 pdf 文本内容"""
    with dst.open("w", encoding="utf-8") as f:
        for i, page in enumerate(load_pdf(src, pages_filter)):
            f.write(f"\n[pdf2text: Page {i + 1}]\n\n")
            for element in page:
                etype: str = type(element).__name__
                # LTTextBox[LTTextLine[LTChar]]
                if etype.startswith("LTTextBox"):
                    for line in element:
                        f.write(line.get_text())
                else:
                    if etype.startswith("LTTextLine") or etype.startswith("LTChar"):
                        raise NotImplementedError(f"Unsupported element: {etype}")


def merge_pdf(src: Iterator[Path], dst: Path):
    import PyPDF2

    merger = PyPDF2.PdfMerger()
    for f in src:
        try:
            merger.append(f)
        except:
            print(f"Failed to merge \"{f}\"")
    merger.write(str(dst))


def pdf2img(file: Path, suffix=".png", root="Project", blowup=15):
    """
    :param file: pdf 文件
    :param suffix: 图像后缀名
    :param root: 保存的源目录
    :param blowup: 图像清晰度
    """
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
    ROOT = Path(r"D:\Downloads\gradebook_CS112-30023448-2025SP_lab204_2025-03-17-12-56-02")
    merge_pdf(ROOT.iterdir(), ROOT.parent / "merge.pdf")
