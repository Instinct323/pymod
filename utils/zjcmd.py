import os


def execute(cmd, check=True):
    ret = print("\033[32m\033[1m" + cmd + "\033[0m") or os.system(cmd)
    if check and ret: raise OSError(f"Fail to execute: {cmd}")
    return ret


def colorstr(msg, *setting):
    setting = ("blue", "bold") if not setting else ((setting,) if isinstance(setting, str) else setting)
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code
    colors = {
        # basic colors
        "black": "\033[30m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
        "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m", "white": "\033[37m",
        # bright colors
        "bright_black": "\033[90m", "bright_red": "\033[91m", "bright_green": "\033[92m",
        "bright_yellow": "\033[93m", "bright_blue": "\033[94m", "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m", "bright_white": "\033[97m",
        # misc
        "end": "\033[0m", "bold": "\033[1m", "underline": "\033[4m",
    }
    return "".join(colors[x] for x in setting) + str(msg) + colors["end"]
