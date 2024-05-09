import datetime
import os
import sys
from pathlib import Path

from zjexe import execute

SCRIPTS = Path(sys.executable).parent / "Scripts"


def git_push(*repositories,
             msg=f"update on {datetime.datetime.today()}"):
    for repo in repositories:
        os.chdir(repo), print(repo.center(50, "-"))
        for cmd in ("git status", "git add .",
                    f"git commit -m \"{msg}\"", "git push origin master"): execute(cmd, check=False)


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
        cls.f.write_text("\n".join(ext)) \
            if ext else cls.f.unlink(missing_ok=True)
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
    def jupyter(cls, root="D:/Workbench", cfg=False, reinstall=False):
        if reinstall:
            tar = ("jupyter", "jupyter-client", "jupyter-console", "jupyter-core",
                   "jupyterlab-pygments", "jupyterlab-widgets", "notebook==6.1.0",
                   "jupyter_contrib_nbextensions")
            for pkg in tar: cls.install(pkg, uninstall=True)
            for pkg in tar: cls.install(pkg)
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
    path = Path("D:/Software/Anaconda3/condabin" if os.name == "nt" else "/opt/miniconda/bin")

    @classmethod
    def add_path(cls):
        if cls.path.is_dir() and str(cls.path) not in os.environ["PATH"]:
            os.environ["PATH"] = str(cls.path) + os.pathsep + os.environ["PATH"]

    @staticmethod
    def create(name, version=(3, 8, 15)):
        version = ".".join(map(str, version))
        execute(f"conda create -n {name} python=={version}")

    @classmethod
    def install(cls, pkg, uninstall=False, upgrade=False):
        # note: 注意 envs 文件夹的权限问题
        main = "uninstall -y" if uninstall else ("upgrade" if upgrade else "install")
        execute(f"conda {main} {pkg}")

    @staticmethod
    def clean():
        execute("conda clean -ay")

    @classmethod
    def config(cls):
        super().config()
        for p in ("--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/",
                  "--add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/",
                  "--set show_channel_urls yes"):
            execute(f"conda config {p}")


if __name__ == "__main__":
    os.chdir(os.getenv("dl"))

    # PythonEnv.install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    # PythonExtLibs.dump([r"D:\Workbench\pymod", r"D:\Workbench\ros_humble\py"])
    git_push("D:/Workbench/cppmod", "D:/Workbench/pymod", "D:/Information/Notes", "D:/Information/Lib")
