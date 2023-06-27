from collections import deque


class PositionalPID:
    ''' 位置式 PID'''

    def __init__(self, kp, ki, kd, target=0.):
        self.kp, self.ki, self.kd, self.target = kp, ki, kd, target
        self.reset()

    def reset(self, n=3):
        # 历史误差 (最近 n 帧), 累积误差 (EMA)
        self._lerror, self._serror = deque(maxlen=n), 0.

    def update(self, measure, m=0.9):
        error = self.target - measure
        self._lerror.append(error)
        self._serror = error + m * self._serror
        derror = error - self._lerror[0]
        return self.kp * error + self.ki * self._serror + self.kd * derror
