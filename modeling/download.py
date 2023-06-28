import os
import re
from pathlib import Path

import requests
from lxml.etree import HTML
from tqdm import tqdm


def math_paper(url, root=Path(os.environ['TMP'])):
    html = HTML(requests.get(url).text)
    title = re.sub(r'\s', '', re.sub(r'[\\/:*?"<>|]', '_', html.xpath('/html/body/div/div[3]/div[1]/div[2]')[0].text))
    # 以文章标题作为路径名, 存放在系统的临时目录下
    root = root / title
    root.mkdir(exist_ok=True)
    # 获取所有图片的源地址
    get_src = lambda node: node.attrib['src']
    nodes = map(lambda node: 'https://dxs.moe.gov.cn' * (not get_src(node).startswith('http')) + get_src(node),
                html.xpath('/html/body/div[1]/div[3]/div[1]/div[4]/div/div[1]/div/div/img'))
    # 开始下载
    for i, src in tqdm(tuple(enumerate(nodes)), desc=str(root.parent)):
        (root / f'{i + 1}.jpg').write_bytes(requests.get(src).content)


if __name__ == '__main__':
    # 中国大学生在线: https://dxs.moe.gov.cn/zx/hd/sxjm/sxjmlw/
    urls_2022a = ['https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2022qgdxssxjmjslwzs/221106/1820295.shtml',
                  'https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2022qgdxssxjmjslwzs/221106/1820293.shtml',
                  'https://dxs.moe.gov.cn/zx/a/hd_sxjm_sxjmlw_2022qgdxssxjmjslwzs/221106/1820291.shtml']
    for url in urls_2022a: math_paper(url)
