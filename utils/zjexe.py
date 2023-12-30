import os
import shutil
import sys
import threading
import time
from ctypes import windll
from pathlib import Path

import psutil

execute = lambda x: print(x) and os.system(x)


class MsgBox:

    @staticmethod
    def info(msg):
        windll.user32.MessageBoxW(0, str(msg), 'info', 0x40)

    @staticmethod
    def warning(msg):
        windll.user32.MessageBoxW(0, str(msg), 'warning', 0x30)

    @staticmethod
    def error(msg):
        windll.user32.MessageBoxW(0, str(msg), 'error', 0x10)
        sys.exit()

    @staticmethod
    def elevate():
        if windll.shell32.IsUserAnAdmin(): return
        windll.shell32.ShellExecuteW(None, 'runas', sys.executable, ' '.join(sys.argv), None, 1)
        sys.exit()


class SingletonExecutor:
    # 根据所运行的 py 文件生成程序标识
    exeid = (Path.cwd() / Path(__file__).name).resolve().as_posix().split(':')[-1].replace('/', '')

    @classmethod
    def check(cls):
        # 通过临时文件, 保证当前程序只在一个进程中被执行
        f = Path(os.getenv('tmp')) / f'py-{cls.exeid}'
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
                raise RuntimeError(f'The current program has been executed in process {other.pid}')
        # 继续: 创建文件描述当前进程
        f.write_text(' '.join(map(str, (cur.pid, cur.create_time()))))

    @classmethod
    def check_daemon(cls, t_wait=1):
        def handler():
            while True:
                cls.check()
                time.sleep(t_wait)

        task = threading.Thread(target=handler, daemon=True)
        task.start()


class Installer:
    cmd = 'pyinstaller'

    @staticmethod
    def add_path():
        from envs import PythonEnv
        PythonEnv.add_path()

    def __init__(self,
                 main: Path,
                 icon: Path = None,
                 paths: list = [],
                 hiddenimports: list = []):
        # 生成工作目录
        wkdir = main.parent / 'install'
        wkdir.mkdir(exist_ok=True)
        os.chdir(wkdir)
        # 记录相关文件
        self.main = main.absolute()
        self.spec = Path(f'{self.main.stem}.spec').absolute()
        self.exclude = Path('exclude.txt').absolute()
        # 记录打包参数
        self.options = []
        if icon:
            self.options.append(f'-i {icon.absolute()}')
        for p in paths:
            self.options.append(f'-p {p}')
        for m in hiddenimports:
            self.options.append(f'--hidden-import {m}')

    def install(self, args='-wF', spec=False):
        opt = self.spec if spec else ' '.join(map(str, self.options + [args, self.main]))
        execute(f'{self.cmd} {opt}')

    def clear(self):
        for f in ('build', 'dist'):
            shutil.rmtree(f, ignore_errors=True)

    def load_exclude(self):
        return self.exclude.read_text().split('\n')

    def dump_exclude(self, fmts=('dll', 'pyd', 'so')):
        # one-dir 打包, 检测依赖项
        self.install('-cD', spec=False)
        input('Verify that the program is running: ')

        src = Path(f'dist/{self.main.stem}/_internal')
        exclude = set()

        for fmt in fmts:
            for f in src.rglob(f'*.{fmt}'):
                try:
                    f.unlink()
                    f = f.relative_to(src)
                    exclude.add(f.as_posix())
                    print('Remove:', f)
                except PermissionError:
                    pass

        exclude = sorted(exclude)
        self.exclude.write_text('\n'.join(map(str, exclude)))

    def __repr__(self):
        return f'{type(self).__name__}({self.main}, options={self.options})'


if __name__ == '__main__':
    Installer.add_path()

    isl = Installer(Path(r'D:/Workbench/Lab/Deal/1215-Best/AX2.0.py'),
                    icon=Path('D:/Information/Video/icons/pika.ico'))
    isl.clear()
    # isl.dump_exclude()
    print(isl)
