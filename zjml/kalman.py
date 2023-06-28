import numpy as np
import scipy.linalg


class KalmenFilter:
    ''' 卡尔曼滤波器 (单目标):
        _std_weight_position: 位置标准差权值
        _std_weight_velocity: 速度标准差权值
        _motion_tran: 状态转移矩阵'''
    _std_weight_position = 1 / 20
    _std_weight_velocity = 1 / 160

    def __init__(self, ndim=4, dt=1):
        # 状态转移矩阵: x, y, a (h / w), h, vx, vy, va, vh
        self._ndim = ndim
        self._motion_tran = np.eye(ndim * 2)
        self._motion_tran[:ndim, ndim:] = dt * np.eye(ndim)
        self._update_mat = np.eye(ndim, ndim * 2)

    def _get_std(self, aspect_radio, gain_pos=1, gain_vel=1):
        # 根据纵横比赋予权值
        weight_pos = gain_pos * self._std_weight_position * aspect_radio
        weight_vel = gain_vel * self._std_weight_velocity * aspect_radio
        return np.array([weight_pos, weight_pos, 1e-2, weight_pos,
                         weight_vel, weight_vel, 1e-5, weight_vel])

    def initiate(self, measurement):
        ''' 位置信息 -> 均值, 协方差'''
        std = self._get_std(measurement[3], gain_pos=2, gain_vel=10)
        # 均值, 协方差
        mean = np.concatenate([measurement, np.zeros_like(measurement)])
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        ''' 预测: 最优估计, 后验估计协方差 -> 先验估计, 先验估计协方差'''
        std = self._get_std(mean[3])
        # 先验估计: x = F @ x
        mean = self._motion_tran @ mean
        # 先验估计协方差: P = F @ P @ F.T + Q
        covariance = self._motion_tran @ covariance @ self._motion_tran.T + np.diag(np.square(std))
        return mean, covariance

    def project(self, mean, covariance):
        ''' 将估计值, 协方差 映射到 检测空间'''
        weight_pos = self._std_weight_position * mean[3]
        std = np.array([weight_pos, weight_pos, 1e-1, weight_pos])
        # Pos and Vel -> Pos: x = H @ x
        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + np.diag(np.square(std))
        return projected_mean, projected_cov

    def update(self, measurement, mean, covariance):
        ''' 更新: 设备测量值, 先验估计, 先验估计协方差 -> 最优估计, 后验估计协方差'''
        projected_mean, projected_cov = self.project(mean, covariance)
        # 计算卡尔曼增益
        kalman_gain = scipy.linalg.cho_solve(
            scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False),
            (covariance @ self._update_mat.T).T, check_finite=False).T
        # 最优估计: x = x + (y - x) @ K.T
        new_mean = mean + (measurement - projected_mean) @ kalman_gain.T
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov


class SingleObjTrack(KalmenFilter):
    ''' 单目标跟踪器
        init: 目标确认丢失, 重新初始化
        track: 目标未丢失则跟踪, 丢失则预测
        mark_miss: 目标暂时丢失, 标记'''
    miss_time_thresh = 5
    confirm_time_thresh = 3

    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self.status = None, None
        self._hits = 0
        self._time_since_update = 0

    def track(self, measurement=None):
        if isinstance(measurement, np.ndarray):
            self._hits, self._time_since_update = self._hits + 1, 0
            if self:
                self.status = self.predict(*self.status)
        else:
            self.mark_miss()
            if self:
                self.status = self.predict(*self.status)
                measurement = self.status[0][:self._ndim]
        if isinstance(measurement, np.ndarray):
            self.status = self.update(measurement, *self.status) \
                if self else self.initiate(measurement)
        return self.status[0]

    def mark_miss(self):
        self._time_since_update += 1
        if self._time_since_update >= self.miss_time_thresh or self._hits < self.confirm_time_thresh: self.init()

    def __bool__(self):
        return isinstance(self.status[0], np.ndarray)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sot = SingleObjTrack()

    # 生成带扰动的 x 坐标的原始轨迹
    np.random.seed(0)
    n = 40
    x = np.linspace(1, 10, n)
    y = np.log(x) + np.random.random(n) * 0.2
    x += np.random.random(n) * 0.3

    # 丢弃部分测量结果
    range_ = 10, n // 2
    detect = [np.array([cx, cy, 1, 1]) for cx, cy in zip(x, y)]
    for i in range(*range_): detect[i] = None
    filt_none = lambda seq: list(filter(lambda i: np.any(i), seq))
    cnt_none = lambda seq: len(seq) - len(filt_none(seq))

    # 单目标跟踪下的 x-y 坐标轨迹
    predict = [sot.track(mea) for mea in detect]
    print(cnt_none(detect), cnt_none(predict))

    plt.plot(x, y, label='origin', color='orange')
    first_none = range_[0] + SingleObjTrack.miss_time_thresh - 1
    predict = np.array(filt_none(predict))[:, :2]
    plt.plot(*predict[:first_none].T, label='track', color='deepskyblue')
    plt.plot(*predict[first_none:].T, color='deepskyblue')
    plt.legend()
    plt.show()
