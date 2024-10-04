import datetime
from pathlib import Path

from zjcmd import *


class RemoteHost:

    def __init__(self, user, ip, port=22):
        self.user = user
        self.ip = ip
        self.port = port

    def ssh(self):
        return f"ssh -p {self.port} {self}"

    def scp(self, src, dst, is_dir: bool):
        return f'scp {"-r" * is_dir} -P {self.port} \"{src}\" {self}:\"{dst}\"'

    def upload(self, src, dst):
        return self.scp(src, dst, Path(src).is_dir())

    def download(self, src, dst, is_dir: bool):
        return self.scp(src, dst, is_dir)

    def __repr__(self):
        return f"{self.user}@{self.ip}"


class GitRepo:

    def __init__(self, path, branch="master", remote="origin"):
        self.path = Path(path)
        assert self.path.is_dir(), f"{self.path} is not a directory."
        self.branch = branch
        self.remote = remote

    def activate(self):
        os.chdir(self.path), print(colorstr(self, "blue", "bold"))
        execute("git status")
        return self

    def add(self, *files):
        execute(f"git add " + (" ".join(files) if files else "."))
        return self

    def commit(self, msg):
        execute(f"git commit -m \"{msg}\"", check=False)
        return self

    def push(self):
        execute(f"git push {self.remote} {self.branch}")
        return self

    def pull(self):
        execute(f"git pull {self.remote} {self.branch}")
        return self

    def __repr__(self):
        return f"GitRepo<{self.path}, {self.branch}, {self.remote}>"


if __name__ == '__main__':
    # myjammy = RemoteHost("root", "127.0.0.1", 30021)
    # print(myjammy.upload(r"D:\Information\Notes", "/home/workbench/tmp"))

    for p in ("D:/Workbench/cppmod", "D:/Workbench/pymod", "D:/Information/Notes", "D:/Information/Lib"):
        repo = GitRepo(p).activate()
        repo.add().commit(f"update on {datetime.datetime.today()}").push()
