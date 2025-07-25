import collections


class PositionalPID:

    def __init__(self, kp, ki, kd, target=0.):
        self.kp, self.ki, self.kd, self.target = kp, ki, kd, target
        self.reset()

    def reset(self, n=3):
        # historical error (last n frames), cumulative error (EMA)
        self._lerror, self._serror = collections.deque(maxlen=n), 0.

    def update(self, measure, m=0.9):
        error = self.target - measure
        self._lerror.append(error)
        self._serror = error + m * self._serror
        return self.kp * error + self.ki * self._serror + self.kd * (error - self._lerror[0])
