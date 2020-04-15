# -*- coding:utf-8 -*-

from __future__ import division, print_function

import numpy as np
from ..estimator import IterationEstimator


class Baseline(IterationEstimator):
    """
    虽然是baseline，不过整体表现比itemcf和slopOne还高，
    也可以看出邻居模型的弊端了，缺少优化目标

    属性
    ---------
    n_factors : 隐式因子数
    n_epochs : 迭代次数
    lr : 学习速率
    reg : 正则因子
    """

    def __init__(self, n_factors=20, n_epochs=20, lr=0.007, reg=.002):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

    def _prepare(self):
        self.user_num = self.train_dataset.matrix.shape[0]
        self.item_num = self.train_dataset.matrix.shape[1]

        self.global_mean = self.train_dataset.global_mean

        # user bias
        self.bu = np.zeros(self.user_num, np.double)

        # item bias
        self.bi = np.zeros(self.item_num, np.double)

    def _iteration(self):
        for u, i, r in self.train_dataset.all_ratings():
            # 预测值
            rp = self.global_mean + self.bu[u] + self.bi[i]
            # 误差
            e_ui = r - rp

            self.bu[u] += self.lr * (e_ui - self.reg * self.bu[u])
            self.bi[i] += self.lr * (e_ui - self.reg * self.bi[i])

    def _pred(self):
        return self.global_mean + np.repeat(np.asmatrix(self.bu).T, self.item_num, axis=1) \
               + np.repeat(np.asmatrix(self.bi), self.user_num, axis=0)

    def predict(self, u, i):
        est = self.global_mean + self.bu[u] + self.bi[i]
        return est