import shutil
import sys
from pathlib import Path

import pip

from zjcmd import *

USERPATH = Path(os.path.expanduser("~"))
SCRIPTS = Path(sys.executable).parent  # Default for Linux
if os.name == "nt": SCRIPTS = SCRIPTS / "Scripts"  # Special for Windows
SITE_PACKAGES = Path(pip.__file__).parent.parent


def elevate():
    from ctypes import windll
    if windll.shell32.IsUserAnAdmin(): return
    # 以管理员身份运行
    windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    sys.exit()


class PythonExtLibs:
    f = SITE_PACKAGES / "extlib.pth"

    @classmethod
    def temp_disable(cls):
        """ Temporarily disable the extension libraries. """
        tmp = cls.f.with_suffix(".bak")
        cls.f.rename(tmp)
        input("Press any key to re-enable...")
        tmp.rename(cls.f)

    @classmethod
    def load(cls):
        return set(
            p for p in cls.f.read_text().splitlines() if p
        ) if cls.f.is_file() else set()

    @classmethod
    def dump(cls, ext):
        ext = [str(p.absolute()) for p in map(Path, ext) if p.is_dir()]
        cls.f.write_text("\n".join(ext)) if ext else cls.f.unlink(missing_ok=True)
        return ext

    @classmethod
    def add(cls, paths):
        ext = cls.load()
        return cls.dump(ext | set(map(str, paths)))

    @classmethod
    def remove(cls, paths):
        ext = cls.load()
        return cls.dump(ext - set(map(str, paths)))


class PythonEnv:
    _pip = SCRIPTS / "pip"
    _jupyter = SCRIPTS / "jupyter"

    def __init__(self):
        if os.name == "nt":
            root = SCRIPTS.parent
            lib = root / "Library"
            for p in (root, SCRIPTS, lib / "bin", lib / "mingw-w64" / "bin", lib / "usr" / "bin"):
                add_path(p)
        else:
            raise NotImplementedError

    @classmethod
    def install(cls, pkg, uninstall=False, upgrade=False):
        # 安装路径: python -m site
        main = "uninstall -y" if uninstall else ("install" + upgrade * " -U")
        execute(f"{cls._pip} {main} --no-cache-dir {pkg}")

    @classmethod
    def load_requirements(cls, file="requirements.txt"):
        execute(f"{cls._pip} install -r {str(file)} -f https://download.pytorch.org/whl/torch_stable.html")

    @classmethod
    def clean(cls):
        if os.name == "nt":
            cache = USERPATH / "AppData" / "Local" / "pip" / "cache"
            shutil.rmtree(cache, ignore_errors=True)
        else:
            raise NotImplementedError

    @classmethod
    def jupyter(cls, root=".", cfg=False):
        # 代码补全: https://blog.csdn.net/qq_55745968/article/details/145530166
        os.chdir(root)
        execute(f"{cls._jupyter} notebook" + cfg * " --generate-config")

    @classmethod
    def config(cls):
        for k, v in (("timeout", 6000),
                     ("index-url", "https://mirrors.aliyun.com/pypi/simple/"),
                     ("trusted-host", "mirrors.aliyun.com")):
            execute(f"{cls._pip} config set global.{k} {v}")
        PythonEnv.install("pip", upgrade=True)


class CondaEnv(PythonEnv):
    _conda = Path("D:/Software/Anaconda3/condabin" if os.name == "nt" else "/opt/miniconda3/bin") / "conda"

    @classmethod
    def create(cls, name, version=(3, 10, 16)):
        version = ".".join(map(str, version))
        execute(f"{cls._conda} create -n {name} python=={version}")

    @classmethod
    def install(cls, pkg, uninstall=False, upgrade=False):
        # note: 注意 envs 文件夹的权限问题
        main = "uninstall -y" if uninstall else ("upgrade" if upgrade else "install")
        execute(f"{cls._conda} {main} {pkg}")

    @classmethod
    def clean(cls):
        super().clean()
        execute(f"{cls._conda} clean -ay")

    @classmethod
    def config(cls):
        super().config()
        for p in ("--set show_channel_urls yes",):
            execute(f"{cls._conda} config {p}")

        for ch in ("https://mirrors.ustc.edu.cn/anaconda/pkgs/free/",
                   "https://mirrors.ustc.edu.cn/anaconda/pkgs/main/",):
            execute(f"{cls._conda} config --add channels {ch}")


def set_print_only():
    PythonEnv._pip = "pip"
    PythonEnv._jupyter = "jupyter"
    CondaEnv._conda = "conda"
    global execute
    execute = print


if __name__ == "__main__":
    set_print_only()
    # PythonExtLibs.dump(["/opt/ros/noetic/lib/python3/dist-packages"])
    # print(PythonExtLibs.load())

    CondaEnv.config()
    # PythonEnv.install("aiohttp")
