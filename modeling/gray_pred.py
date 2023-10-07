import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import LOGGER

warnings.filterwarnings('ignore')


def gray_correlation(refer, data, rho=0.5):
    ''' :param refer: 参照数列 (列向量)
        :param data: 比较数列 (以列为单位)
        :param rho: 分辨率
        :return: 灰色关联度'''
    # 确保参照数列为列向量
    refer = refer.reshape(-1, 1)
    # 数列间的绝对距离
    distance = np.abs(data - refer)
    dis_min = distance.min()
    dis_max = rho * distance.max()
    # 关联系数: 比较数列中每个值与参照数列的关联性
    cor_param = (dis_min + dis_max) / (distance + dis_max)
    # 关联度: 关联系数按列求平均
    degree = cor_param.mean(axis=0)
    return degree


def gray_poly(seq, alpha=0.5):
    ''' :return: 灰多项式 (加权邻值生成数列)'''
    part_1 = alpha * seq[1:]
    part_2 = (1 - alpha) * seq[:-1]
    return part_1 + part_2


def gray_model(seq, alpha=0.5, decimals=4, show=False):
    ''' 灰色预测模型建立
        :param seq: 原始序列
        :param alpha: 灰多项式权值
        :param decimals: 数值精度
        :param show: 绘制真实值和预测值的对比图
        :return: 灰色预测模型, 模型信息'''
    seq, n = seq.flatten(), seq.size
    index = np.arange(1, n + 1)
    # 创建数据存储表单
    model_info = pd.DataFrame(index=index, columns=['真实值', '模型值', '相对误差', '级比偏差'])
    model_info['真实值'] = seq
    # 级比: x(k-1) / x(k)
    mag_ratio = seq[:-1] / seq[1:]
    mr_r = round(mag_ratio.max(), decimals)
    mr_l = round(mag_ratio.min(), decimals)
    # 级比检验的区间边界
    l = round(np.exp(- 2 / (n + 1)), decimals)
    r = round(np.exp(2 / (n + 1)), decimals)
    # 级比检验: 检验数列级比是否都落在可容覆盖区间
    LOGGER.info('\n'.join(['级比检验:', f'MR ∈ [{mr_l}, {mr_r}]',
                           f'Border ∈ [{l}, {r}]', '']))
    assert mr_l >= l and mr_r <= r, '序列未通过级比检验, 可通过平移变换改善'
    # 灰积分: 累加数列
    integrade = np.cumsum(seq)
    # 灰微分: 差分数列
    diff = np.diff(integrade)
    # 灰多项式: 加权邻值生成数列
    white = alpha * integrade[1:] + (1 - alpha) * integrade[:-1]
    # 求解灰微分方程: diff + a * white = b
    hidden = np.stack([-white, np.ones(white.shape)], axis=0)
    # shape: [2, n] × [n, 2] × [2, n] × [n, ] -> [2, ]
    a, b = (np.linalg.inv(hidden @ hidden.T) @ hidden @ diff)
    LOGGER.info('\n'.join([f'发展灰度: {a}', f'内生控制灰度: {b}']))
    c2 = b / a
    c1 = seq[0] - c2

    def inte_pred(time):
        ''' 白化模型预测函数
            :param time: 序列首个值对应的时间为 1'''
        assert np.all(time >= 1)
        return c1 * (np.exp(- a * (time - 1)) - np.exp(- a * (time - 2)))

    LOGGER.info(
        f'预测方程: x(t) = {round(c1, decimals)}·[e^({-round(a, decimals)}(t-1)) - e^({-round(a, decimals)}(t-2))]\n')
    # 得到白化模型预测值
    pred = inte_pred(index)
    model_info['模型值'] = np.round(pred, decimals)
    # 计算得到误差信息
    model_info['相对误差'] = np.round(np.abs((seq - pred) / seq), decimals)
    model_info['级比偏差'] = np.concatenate([np.zeros(1), np.round(
        np.abs(1 - (1 - 0.5 * a) / (1 + 0.5 * a) * mag_ratio), decimals)])
    # 相对残差 / 级比偏差: < 0.1 达到较高要求, < 0.2 达到一般要求
    for label in ['相对误差', '级比偏差']:
        error = model_info[label]
        if np.all(error < 0.1):
            message = 'Excelent'
        elif np.all(error < 0.2):
            message = 'Common'
        else:
            raise AssertionError(f'{label}: Fail {error.max()}')
        LOGGER.info(f'{label}: {message}')
    # 输出模型信息
    LOGGER.info('')
    LOGGER.info(model_info)
    if show:
        # 绘制真实值和预测值的对比图
        plt.plot(index, model_info['真实值'], c='deepskyblue', label='true')
        plt.scatter(index, model_info['模型值'], c='orange', label='pred', marker='p')
        plt.legend()
        plt.show()
    return inte_pred, model_info
