import pickle
from pathlib import Path

import fitz
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import *
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from tqdm import tqdm

STOP_WORDS = r'\/:*?"<>|'


def seq_comp(seq1, seq2):
    ''' 序列匹配度计算'''
    n1, n2 = map(len, (seq1, seq2))
    dp = [[int(e1 == e2) for e2 in seq2] for e1 in seq1]
    for c in range(1, n2): dp[0][c] = max(dp[0][c], dp[0][c - 1])
    for r in range(1, n1):
        dp[r][0] = max(dp[r][0], dp[r - 1][0])
        for c in range(1, n2):
            dp[r][c] = dp[r - 1][c - 1] + 1 \
                if dp[r][c] else max(dp[r - 1][c], dp[r][c - 1])
    return dp[-1][-1] / max(n1, n2)


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


def pdf2img(file: Path, suffix='.jpg', root='Project', blowup=15):
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


if __name__ == '__main__':
    pass
