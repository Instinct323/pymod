import os
import re
from pathlib import Path

import requests
from lxml.etree import HTML
from tqdm import tqdm


def math_paper(url, path=Path(os.environ['TMP'])):
    html = HTML(requests.get(url).text)
    title = re.sub(r'\s', '', re.sub(r'[\\/:*?"<>|]', '_', html.xpath('/html/body/div/div[3]/div[1]/div[2]')[0].text))
    # 以文章标题作为路径名, 存放在系统的临时目录下
    root = path / title
    root.mkdir(exist_ok=True)
    # 获取所有图片的源地址
    get_src = lambda node: node.attrib['src']
    nodes = map(lambda node: 'https://dxs.moe.gov.cn' * (not get_src(node).startswith('http')) + get_src(node),
                html.xpath('/html/body/div[1]/div[3]/div[1]/div[4]/div/div[1]/div/div/img'))
    # 开始下载
    pbar = tqdm(tuple(enumerate(nodes)), desc=str(root.parent))
    for i, src in pbar:
        (root / f'{i + 1}.jpg').write_bytes(requests.get(src).content)


if __name__ == '__main__':
    urls_2021b = ['https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2021qgdxssxjmjslwzs/211024/1734072.shtml',
                  'https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2021qgdxssxjmjslwzs/211026/1734070.shtml',
                  'https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2021qgdxssxjmjslwzs/211024/1734074.shtml',
                  'https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2021qgdxssxjmjslwzs/211025/1734083.shtml']
    for url in urls_2021b: math_paper(url)
