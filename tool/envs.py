import os
import shutil
import sys
from ctypes import windll
from pathlib import Path

from zjcmd import execute

USERPATH = Path(os.path.expanduser("~"))
SCRIPTS = Path(sys.executable).parent  # Default for Linux
if os.name == "nt": SCRIPTS = SCRIPTS / "Scripts"  # Special for Windows


def elevate():
    if windll.shell32.IsUserAnAdmin(): return
    # 以管理员身份运行
    windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    sys.exit()


class PythonExtLibs:
    f = Path(sys.executable).parent / "extlib.pth"

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

    @classmethod
    def install(cls, pkg, uninstall=False, upgrade=False):
        # 安装路径: python -m site
        main = "uninstall -y" if uninstall else ("install" + upgrade * " --upgrade")
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
    def jupyter(cls, root="D:/Workbench", cfg=False, reinstall=False):
        if reinstall:
            tar = ("jupyter", "jupyter-client", "jupyter-console", "jupyter-core",
                   "jupyterlab-pygments", "jupyterlab-widgets", "notebook",
                   "jupyter_contrib_nbextensions")
            for pkg in tar: cls.install(pkg, uninstall=True)
            cls.install("notebook==6.1.0")
            cls.install("jupyter")
            execute(f"{cls._jupyter} contrib nbextension install --use")

        os.chdir(root)
        execute(f"{cls._jupyter} notebook" + cfg * " --generate-config")

    @classmethod
    def config(cls):
        for k, v in (("timeout", 6000),
                     ("index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"),
                     ("trusted-host", "pypi.tuna.tsinghua.edu.cn")):
            execute(f"{cls._pip} config set global.{k} {v}")


class CondaEnv(PythonEnv):
    _conda = Path("D:/Software/Anaconda3/condabin" if os.name == "nt" else "/opt/miniconda/bin") / "conda"

    @classmethod
    def create(cls, name, version=(3, 8, 15)):
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
        for p in ("--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/",
                  "--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/",
                  "--set show_channel_urls yes"):
            execute(f"{cls._conda} config {p}")


if __name__ == "__main__":
    os.chdir(os.getenv("dl"))
    # elevate()

    PythonEnv.jupyter()
