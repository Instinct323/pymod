import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

# 普通左对齐文本html设置: 12 号左对齐黑体
common_l = lambda text: f'<html><head/><body><p><span style=" font-size:12pt; ' \
                        f'font-weight:600;">{text}</span></p></body></html>'
# 标题左对齐文本html设置: 12 号左对齐加粗蓝色
head = lambda text: f'<html><head/><body><p><span style=" font-size:12pt; ' \
                    f'font-weight:600; cor:#0da7ef;">{text}</span></p></body></html>'


def select_file(handler, glob_pats=('*.*',), w_app=True):
    from PyQt5.QtWidgets import QFileDialog, QApplication
    # 请执行 sys.exit(0) 退出程序
    if not w_app: app = QApplication(sys.argv)
    dialog = QFileDialog()
    file = dialog.getOpenFileName(caption='Select file',
                                  filter=('; '.join(glob_pats).join('()') if glob_pats else None))[0]
    if file: return handler(Path(file))


def select_dir(handler, w_app=True):
    from PyQt5.QtWidgets import QFileDialog, QApplication
    # 请执行 sys.exit(0) 退出程序
    if not w_app: app = QApplication(sys.argv)
    dialog = QFileDialog()
    dire = dialog.getExistingDirectory(None, 'Select directory')
    if dire: return handler(Path(dire))


class PltFigure(plt.Figure):

    def __init__(self):
        super().__init__()
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        self.widget = FigureCanvasQTAgg(self)
        # 函数重命名
        self.show = self.canvas.draw

    def subplot(self, *args, **kwargs) -> plt.Axes:
        return self.add_subplot(*args, **kwargs)


class Window(QMainWindow):

    def __init__(self, opacity=1.):
        super().__init__()
        self.central = QWidget()
        self.setCentralWidget(self.central)
        # 只显示关闭按钮
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowOpacity(opacity)
        # 窗口尺寸
        self.setFixedSize(720, 540)
        self.setWindowTitle('Title')
        # 状态栏, 菜单栏
        self.setStatusBar(QStatusBar())
        self.setMenuBar(QMenuBar())

        file = self.menuBar().addAction('file')

        # matplot
        self.fig = PltFigure()

        # 调用布局函数, 并连接信号与槽
        self.layout()
        file.triggered.connect(partial(select_file, print))

    def plot(self):
        self.fig.clear()
        # 在此进行绘图
        for i in range(4):
            fig = self.fig.subplot(2, 2, i + 1)
            x = np.random.normal(0, 1, 1000)
            fig.hist(x, bins=50, color='orange')
        # ---------------
        self.fig.show()

    def message(self, msg):
        self.statusBar().showMessage(msg, int(1e8))

    def layout(self):
        lay = QVBoxLayout(self.central)
        lay.addWidget(self.fig.widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.plot()
    window.show()
    sys.exit(app.exec_())
