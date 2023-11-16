import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

# 普通左对齐文本html设置: 12 号左对齐黑体
common_l = lambda text: f'<html><head/><body><p><span style=" font-size:12pt; ' \
                        f'font-weight:600;">{text}</span></p></body></html>'
# 标题左对齐文本html设置: 12 号左对齐加粗蓝色
head = lambda text: f'<html><head/><body><p><span style=" font-size:12pt; ' \
                    f'font-weight:600; cor:#0da7ef;">{text}</span></p></body></html>'


class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.central = QWidget()
        self.setCentralWidget(self.central)
        # 初始化状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.setWindowOpacity(0.85)
        # 只显示关闭按钮
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        # 窗口尺寸
        self.setFixedSize(720, 540)
        self.setWindowTitle('焦点逻辑回归')
        # self.status.showMessage('', int(1e8))

        self.layout()

    def layout(self):
        self.lay = QVBoxLayout(self.central)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
