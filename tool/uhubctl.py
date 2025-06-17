import os
import time

from zjexe import execute


def get_idle_time():
    return int(os.popen("xprintidle").read()) / 1e3


def plot_idle_time(freq: float = 1.,
                   duration: float = 60.):
    import matplotlib.pyplot as plt
    states = []
    while True:
        states.append(get_idle_time())
        if len(states) > duration * freq: states.pop(0)

        plt.cla()
        plt.plot(states, color="deepskyblue", linewidth=2)
        plt.grid()
        plt.xlabel("Time (s)")
        plt.ylabel("Idle Time (ms)")
        plt.pause(1 / freq)


def usb_auto_control(argv: str,
                     idle_time_thresh: float,
                     freq: float = 1.):
    """
    Automatically control USB power based on idle time.
    :param argv: Arguments for uhubctl command, e.g., "-l 1-1 -p 2"
    :param idle_time_thresh: Idle time threshold in seconds to turn off USB power
    :param freq: Frequency in Hz to check idle time
    """
    power = False
    while True:
        time.sleep(1 / freq)
        is_idle = get_idle_time() > idle_time_thresh

        if power == is_idle:
            action = "off" if is_idle else "on"
            execute(f"uhubctl {argv} -a {action}")
            power = not power


if __name__ == '__main__':
    # plot_idle_time()
    usb_auto_control(300)
