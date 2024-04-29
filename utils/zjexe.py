import os
import shutil
import sys
import time
from pathlib import Path

import psutil


def execute(cmd, check=True):
    ret = print(cmd) or os.system(cmd)
    if check and ret: raise OSError(f"Fail to execute: {cmd}")


def find_exe(name):
    for p in psutil.process_iter():
        if p.name() == name: return p


def pyc2py(pyc):
    pyc = Path(pyc)
    execute(f"uncompyle6 {pyc} > {pyc.with_suffix('.py')}")


class Installer:
    """ cite: https://blog.csdn.net/qq_55745968/article/details/135430884

        :param main: 主程序文件
        :param console: 是否显示控制台
        :param icon: 图标文件
        :param paths: 搜索路径 (非必需)
        :param hiddenimports: 导入模块 (非必需)

        :ivar opt_mode: 模式参数 (不适用于 spec)
        :ivar opt_general: 通用的参数"""
    exe = Path(sys.executable).parent / "Scripts" / "pyinstaller"

    def __init__(self,
                 main: Path,
                 console: bool = True,
                 icon: Path = None,
                 paths: list = [],
                 hiddenimports: list = []):
        # 生成工作目录
        wkdir = main.parent / "install"
        wkdir.mkdir(exist_ok=True)
        os.chdir(wkdir)
        # 记录相关文件
        self.main = main.absolute()
        self.spec = Path(f"{self.main.stem}.spec").absolute()
        self.exclude = Path("exclude.txt").absolute()
        # 模式参数 (不适用于 spec)
        self.opt_mode = ["-c" if console else "-w"]
        if icon:
            self.opt_mode.append(f"-i {icon.absolute()}")
        # 通用的参数
        self.opt_general = []
        for p in paths:
            self.opt_general.append(f"-p {p}")
        for m in hiddenimports:
            self.opt_general.append(f"--hidden-import {m}")

    def install(self, one_file=True, spec=False):
        """ :param one_file: 单文件打包 / 多文件打包
            :param spec: 使用 spec 文件打包"""
        opt_mode = " ".join(self.opt_mode + ["-F" if one_file else "-D"])
        opt_general = " ".join(self.opt_general)
        target = self.spec if spec else (opt_mode + " " + str(self.main))
        execute(f"{self.exe} {opt_general} {target}")

    def clean(self, build=False, dist=True):
        fs = []
        if build: fs.append("build")
        if dist: fs.append("dist")
        fs = list(map(Path, fs))
        # 尝试删除
        while True:
            # 关闭正在运行的程序
            exe = find_exe(self.main.stem + ".exe")
            if dist and exe:
                print("Kill:", exe.name())
                exe.kill(), time.sleep(1)
            try:
                for f in fs:
                    if f.is_dir():
                        shutil.rmtree(f, ignore_errors=False)
                break
            except PermissionError as e:
                input(f"\n{e}\nPlease resolve the above error: ")

    def load_exclude(self):
        return self.exclude.read_text().split("\n")

    def dump_exclude(self, fmts=("dll", "pyd", "so")):
        # 依赖文件所在文件夹 (根据版本确定)
        src = Path(f"dist/{self.main.stem}/_internal")
        import PyInstaller
        if int(PyInstaller.__version__[0]) == 5: src = src.parent
        # one-dir 打包
        self.install(one_file=False, spec=False)
        # 确保程序正在运行
        while not find_exe(self.main.stem + ".exe"):
            input("Verify that the program is running: ")
        # 尝试删除依赖项
        exclude = []
        for fmt in fmts:
            for f in src.rglob(f"*.{fmt}"):
                try:
                    f.unlink()
                    f = f.relative_to(src)
                    exclude.append(f.as_posix())
                    print("Remove:", f)
                except PermissionError:
                    pass
        # 写入 exclude.txt
        exclude.sort()
        self.exclude.write_text("\n".join(map(str, exclude)))

    def modify_spec(self):
        with self.spec.open("r") as f:
            lines = f.readlines()
        if not lines[2].startswith("# todo: "):
            # 在第 3 行插入代码
            for i, code in enumerate(
                    ("# todo: Loads the list of excluded files",
                     "from pathlib import Path",
                     "EXE.my_exclude = Path('exclude.txt').read_text().splitlines()")):
                lines.insert(i + 2, code + "\n")
            # 保存文件
            with self.spec.open("w") as f:
                f.writelines(lines)

    def __repr__(self):
        return f"{type(self).__name__}({self.main}, opt_general={self.opt_general}, opt_mode={self.opt_mode})"

    @staticmethod
    def check_src(mark="if typecode in ('BINARY', 'EXTENSION'):"):
        from pathlib import Path
        from PyInstaller.building import api
        file = Path(api.__file__)

        with file.open("r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                # 查找代码的插入位置
                if lines[i].strip() == mark:
                    # 查找是否已插入代码
                    for j in range(i + 1, i + 10):
                        if "getattr(EXE, 'my_exclude', [])" in lines[j]: return True

                    # 未插入代码
                    sep = "-" * 15
                    content = [rf"{i + 1:<8d}{mark}",
                               rf"{i + 2:<12d}# {sep} ↓ INSERT ↓ {sep}",
                               rf"{i + 3:<12d}if dest_name.replace('\\', '/') in getattr(EXE, 'my_exclude', []):",
                               rf"{i + 4:<16d}print('Skip:', dest_name)",
                               rf"{i + 5:<16d}continue",
                               rf"{i + 6:<12d}# {sep} ↑ INSERT ↑ {sep}"]
                    content = "\n".join(content)
                    raise RuntimeError(f"Please modify {file} first\n\n{content}")

        # 版本要求: 5.8.0 以上
        import PyInstaller
        raise RuntimeError(f"Fail to solve PyInstaller {PyInstaller.__version__}")


if __name__ == "__main__":
    # website: https://upx.github.io/
    upx_dir = None  # "D:/Software/_tool/upx"

    # 校验源代码的修改情况, 否则提供修改建议
    print(Installer.__doc__, "\n")
    Installer.check_src()
    isl = Installer(Path("D:/Workbench/Repository/Deal/pyinstaller/__exp__/zjqt.py"),
                    console=False,
                    icon=Path("D:/Information/Source/icon/pika.ico"))

    # note: 随版本迭代, 代码与视频有所不符, 不需要像视频演示那样一步一步走
    # note: 配置好上面几个路径, 直接运行就可以
    isl.clean()
    # Step 1: one-dir 打包, 生成 exclude.txt
    if not isl.exclude.is_file():
        isl.dump_exclude()
    # Step 2: one-file 打包, 生成 spec 文件
    if isl.load_exclude():
        isl.install(one_file=True)
        # Step 3: 修改 spec 文件, 生成最终的 exe 文件
        isl.clean(build=True)
        if upx_dir: isl.opt_general.append(f"--upx-dir {upx_dir}")
        isl.modify_spec()
        isl.install(spec=True)
