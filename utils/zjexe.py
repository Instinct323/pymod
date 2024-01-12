import os
import shutil
import sys
import threading
import time
from ctypes import windll
from pathlib import Path

import psutil


def execute(cmd, check=True):
    ret = print(cmd) or os.system(cmd)
    if check and ret: raise OSError(f'Fail to execute: {cmd}')


def find_exe(name):
    for p in psutil.process_iter():
        if p.name() == name: return p


def pyc2py(pyc):
    pyc = Path(pyc)
    execute(f'uncompyle6 {pyc} > {pyc.with_suffix(".py")}')


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
    # 破解: pyinstxtractor, uncompyle6
    exe = 'pyinstaller'

    @staticmethod
    def check_src(mark="if typecode in ('BINARY', 'EXTENSION'):"):
        from pathlib import Path
        from PyInstaller.building import api
        file = Path(api.__file__)

        with file.open('r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                # 查找代码的插入位置
                if lines[i].strip() == mark:
                    # 查找是否已插入代码
                    for j in range(i + 1, i + 10):
                        if "getattr(EXE, 'my_exclude', [])" in lines[j]: return True

                    # 未插入代码
                    sep = '-' * 15
                    content = [rf"{i + 1:<8d}{mark}",
                               rf"{i + 2:<12d}# {sep} ↓ INSERT ↓ {sep}",
                               rf"{i + 3:<12d}if dest_name.replace('\\', '/') in getattr(EXE, 'my_exclude', []):",
                               rf"{i + 4:<16d}print('Skip:', dest_name)",
                               rf"{i + 5:<16d}continue",
                               rf"{i + 6:<12d}# {sep} ↑ INSERT ↑ {sep}"]
                    content = '\n'.join(content)
                    raise RuntimeError(f'Please modify {file} first\n\n{content}')

        # 版本要求: 5.8.0 以上
        import PyInstaller
        raise RuntimeError(f'Fail to solve PyInstaller {PyInstaller.__version__}')

    def __init__(self,
                 main: Path,
                 console: bool = True,
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
        self.options = ['-c' if console else '-w']
        if icon:
            self.options.append(f'-i {icon.absolute()}')
        for p in paths:
            self.options.append(f'-p {p}')
        for m in hiddenimports:
            self.options.append(f'--hidden-import {m}')

    def install(self, one_file=True, spec=False):
        opt = self.spec if spec else ' '.join(map(str, self.options + ['-F' if one_file else '-D', self.main]))
        execute(f'{self.exe} {opt}')

    def clear(self, build=True, dist=True):
        fs = []
        if build: fs.append('build')
        if dist: fs.append('dist')
        fs = list(map(Path, fs))
        # 尝试删除
        while True:
            # 关闭正在运行的程序
            exe = find_exe(self.main.stem + '.exe')
            if dist and exe:
                print('Kill:', exe.name())
                exe.kill(), time.sleep(1)
            try:
                for f in fs:
                    if f.is_dir():
                        shutil.rmtree(f, ignore_errors=False)
                break
            except PermissionError as e:
                input(f'\n{e}\nPlease resolve the above error: ')

    def load_exclude(self):
        return self.exclude.read_text().split('\n')

    def dump_exclude(self, fmts=('dll', 'pyd', 'so')):
        # 依赖文件所在文件夹 (根据版本确定)
        src = Path(f'dist/{self.main.stem}/_internal')
        import PyInstaller
        if int(PyInstaller.__version__[0]) == 5: src = src.parent
        # one-dir 打包
        self.install(one_file=False, spec=False)
        # 确保程序正在运行
        while not find_exe(self.main.stem + '.exe'):
            input('Verify that the program is running: ')
        # 尝试删除依赖项
        exclude = []
        for fmt in fmts:
            for f in src.rglob(f'*.{fmt}'):
                try:
                    f.unlink()
                    f = f.relative_to(src)
                    exclude.append(f.as_posix())
                    print('Remove:', f)
                except PermissionError:
                    pass
        # 写入 exclude.txt
        exclude.sort()
        self.exclude.write_text('\n'.join(map(str, exclude)))

    def modify_spec(self):
        with self.spec.open('r') as f:
            lines = f.readlines()
        # 在第 3 行插入代码
        for i, code in enumerate(
                ('# todo: Loads the list of excluded files',
                 'from pathlib import Path',
                 'EXE.my_exclude = Path(\'exclude.txt\').read_text().splitlines()')):
            lines.insert(i + 2, code + '\n')
        # 保存文件
        with self.spec.open('w') as f:
            f.writelines(lines)

    def __repr__(self):
        return f'{type(self).__name__}({self.main}, options={self.options})'


if __name__ == '__main__':
    from envs import PythonEnv

    PythonEnv.add_path()

    # 校验源代码的修改情况, 否则提供修改建议
    Installer.check_src()
    isl = Installer(Path('D:/Workbench/Repository/pyinstaller/__exp__/zjqt.py'),
                    console=False,
                    icon=Path('D:/Information/Source/icon/pika.ico'))

    # Step 1: one-dir 打包, 生成 exclude.txt
    isl.clear(build=False)
    isl.dump_exclude()
    if isl.load_exclude():
        # Step 2: one-file 打包, 生成 spec 文件
        isl.clear(build=False)
        isl.install(one_file=True)
        # Step 3: 修改 spec 文件, 生成最终的 exe 文件
        isl.clear()
        isl.modify_spec()
        isl.install(spec=True)
