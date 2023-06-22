import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *


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
