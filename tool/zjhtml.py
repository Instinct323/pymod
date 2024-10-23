import re
from pathlib import Path

import cv2
from lxml import etree


def shrink_img(src, dst, width=720, tag="img", encoding="utf-8"):
    """Resize image to width."""
    src = Path(src)
    root = src.parent
    text = src.read_text(encoding="utf-8")
    # find all images
    for img in etree.HTML(text).xpath(f"//{tag}"):
        h, w, c = cv2.imread(str(root / img.attrib["src"])).shape
        if w > width:
            img.attrib["width"] = str(width)
            img.attrib["height"] = str(int(h * width / w))
            # replace image
            f = img.attrib["src"]
            text = re.sub(rf'<{tag}.*src="{f}".*/>',
                          etree.tostring(img, encoding=encoding).decode(encoding), text)
    Path(dst).write_text(text, encoding=encoding)


if __name__ == '__main__':
    root = Path(r"D:\Workbench\Information\Advanced Nonlinear Optimization\hw5\report")

    shrink_img(root / "hw5.html", root / "hw5.html")
