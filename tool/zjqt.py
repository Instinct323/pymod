import os
import sys
import threading
import time
from functools import partial
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import psutil
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import *

# 普通左对齐文本 html 设置: 左对齐黑体
common_l = lambda text: f"<html><head/><body><p><span style=\" font-size:12pt; " \
                        f"font-weight:600;\">{text}</span></p></body></html>"
# 标题左对齐文本 html 设置: 左对齐加粗蓝色
head = lambda text: f"<html><head/><body><p><span style=\" font-size:12pt; " \
                    f"font-weight:600; cor:#0da7ef;\">{text}</span></p></body></html>"

try:
    class MsgBox:
        from ctypes import windll

        @classmethod
        def info(cls, msg):
            cls.windll.user32.MessageBoxW(0, str(msg), "info", 0x40)

        @classmethod
        def warning(cls, msg):
            cls.windll.user32.MessageBoxW(0, str(msg), "warning", 0x30)

        @classmethod
        def error(cls, msg):
            cls.windll.user32.MessageBoxW(0, str(msg), "error", 0x10)
            sys.exit()
except ImportError:
    pass


class SingletonExecutor:
    # 根据所运行的 py 文件生成程序标识
    exeid = (Path.cwd() / Path(__file__).name).resolve().as_posix().split(":")[-1].replace("/", "")

    @classmethod
    def check(cls):
        # 通过临时文件, 保证当前程序只在一个进程中被执行
        f = Path(os.getenv("tmp")) / f"py-{cls.exeid}"
        # 读取文件, 并判断是否已有进程存在
        cur = psutil.Process()
        if f.is_file():
            try:
                _pid, time = f.read_text().split()
                # 检查: 文件是否描述了其它进程
                assert _pid != str(cur.pid)
                other = psutil.Process(int(_pid))
            except:
                other, time = cur, cur.create_time() + 1
            # 退出: 文件所描述的进程仍然存在
            if other.create_time() == float(time):
                raise RuntimeError(f"The current program has been executed in process {other.pid}")
        # 继续: 创建文件描述当前进程
        f.write_text(" ".join(map(str, (cur.pid, cur.create_time()))))

    @classmethod
    def check_daemon(cls, t_wait=1):
        def handler():
            while True:
                cls.check()
                time.sleep(t_wait)

        task = threading.Thread(target=handler, daemon=True)
        task.start()


def select_file(handler, glob_pats=("*.*",), w_app=True):
    from PyQt5.QtWidgets import QFileDialog, QApplication
    # 请执行 sys.exit(0) 退出程序
    if not w_app: app = QApplication(sys.argv)
    dialog = QFileDialog()
    file = dialog.getOpenFileName(caption="Select file",
                                  filter=("; ".join(glob_pats).join("()") if glob_pats else None))[0]
    if file: return handler(Path(file))


def select_dir(handler, w_app=True):
    from PyQt5.QtWidgets import QFileDialog, QApplication
    # 请执行 sys.exit(0) 退出程序
    if not w_app: app = QApplication(sys.argv)
    dialog = QFileDialog()
    dire = dialog.getExistingDirectory(None, "Select directory")
    if dire: return handler(Path(dire))


class PltFigure(plt.Figure):

    def __init__(self):
        super().__init__()
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        self.widget = FigureCanvasQTAgg(self)
        # redirect show function
        self.show = self.canvas.draw

    def subplot(self, *args, **kwargs) -> plt.Axes:
        return self.add_subplot(*args, **kwargs)


class VideoPlayer(QVBoxLayout):
    frame_id = property(lambda self: self.frame_slider.value())
    frame_count = property(lambda self: int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video else 0)
    fps = property(lambda self: self.video.get(cv2.CAP_PROP_FPS) if self.video else 0)

    def __init__(self,
                 play_shortcut: str = "Space"):
        super().__init__()
        self.video: cv2.VideoCapture = None
        self.canvas = PltFigure()

        # frame display
        self.play_state = False
        self.play_btn = QPushButton()
        self.play_btn.setEnabled(False)
        self.play_btn.setText("Video Player")
        self.play_icons = [self.play_btn.style().standardIcon(k) for k in (QStyle.SP_MediaPlay, QStyle.SP_MediaPause)]
        self.play_btn.clicked.connect(self.switch_state)
        if play_shortcut: QShortcut(QKeySequence(play_shortcut), self.play_btn, activated=self.switch_state)

        # frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.refresh_frame)
        f_pause = partial(self.switch_state, play_state=False)
        self.frame_slider.sliderPressed.connect(f_pause)
        self.frame_slider.actionTriggered.connect(f_pause)

        # timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.increment_frame)

        # setup layout
        self._setup_layout()

    def _setup_layout(self):
        lay_h1 = QHBoxLayout()
        self.addWidget(self.canvas.widget, 10)
        self.addLayout(lay_h1, 1)

        lay_h1.addWidget(self.play_btn, 1)
        lay_h1.addWidget(self.frame_slider, 5)

    def increment_frame(self):
        """ Increment frame by one. """
        frame_id = self.play_state[0] + int((time.time() - self.play_state[1]) * self.fps)
        self.frame_slider.setValue(frame_id) if frame_id < self.frame_count else self.switch_state(play_state=False)

    def switch_state(self, event=None,
                     play_state: bool = None):
        """ Switch play/pause state. """
        self.play_state = not self.play_state if play_state is None else play_state
        self.play_btn.setIcon(self.play_icons[self.play_state])

        if self.play_state: self.play_state = self.frame_id, time.time()
        self.timer.start(10) if self.play_state else self.timer.stop()

    def load_file(self,
                  file: str) -> str:
        """ Load video file. """
        self.video = cv2.VideoCapture(file)
        if not self.video.isOpened():
            return f"Cannot open video file: {file}"

        # reset player
        self.switch_state(play_state=False)
        self.play_btn.setEnabled(True)
        self.frame_slider.setEnabled(True)
        self.frame_slider.setValue(0)
        self.frame_slider.setRange(0, self.frame_count - 1)
        self.refresh_frame()

    def refresh_frame(self, event=None):
        """ Refresh the current frame. """
        fid = self.frame_id
        self.play_btn.setText(f"{fid}/{self.frame_count - 1}")

        self.video.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = self.video.read()

        self.canvas.clear()
        self.canvas.subplot().imshow(frame[..., ::-1])
        self.canvas.show()


class Window(QMainWindow):

    def __init__(self,
                 title: str,
                 win_size: tuple,
                 opacity: float = 1.,
                 icon: str = None):
        super().__init__()
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle(title)
        self.resize(*win_size)
        self.setWindowOpacity(opacity)
        if icon: self.setWindowIcon(QIcon(icon))

        # status bar, menu bar
        self.setStatusBar(QStatusBar())
        self.setMenuBar(QMenuBar())
        file = self.menuBar().addAction("file")

        # matplotlib
        self.player = VideoPlayer()

        # setup layout, connect signals
        self._setup_layout()
        file.triggered.connect(partial(select_file, print))

    def message(self, msg):
        if msg: self.statusBar().showMessage(msg, int(1e8))

    def _setup_layout(self):
        lay = QVBoxLayout(self.central)
        lay.addLayout(self.player)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window("Test", (1080, 720))
    window.show()
    window.player.load_file(r"D:\Workbench\assets\黑羽快斗.mp4")
    sys.exit(app.exec_())
