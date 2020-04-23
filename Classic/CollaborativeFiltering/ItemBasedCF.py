# -*- coding:utf-8 -*-

from __future__ import division, print_function

import numpy as np
from scipy.sparse import lil_matrix
from ..estimator import Estimator
from .baseline import Baseline

#模型继承评估类，可以直接使用评估类理的方法

class Itemcf(Estimator):
    """
    属性
    ---------
    min : 有效交互数下限
    topk : 相似矩阵topk
    use_baseline : 是否嵌入baseline计算bias
    """

    def __init__(self, min=2, topk=20, use_baseline=True):
        self.min = min
        self.topk = topk
        self.use_baseline = use_baseline

    def compute_cosine_similarity(self, user_num, item_num, users_ratings):
        print(user_num, item_num)

        sss

        sim = lil_matrix((item_num, item_num), dtype=np.double)

        #点积
        dot = lil_matrix((item_num, item_num), dtype=np.double)

        #左向量平方和
        sql = lil_matrix((item_num, item_num), dtype=np.double)

        #右向量平方和
        sqr = lil_matrix((item_num, item_num), dtype=np.double)

        #共现矩阵
        coo = lil_matrix((item_num, item_num), dtype=np.double)

        cur = 1
        for u, (ii, rr) in users_ratings:
            cur = cur + 1
            for k in range(len(ii) - 1):
                k1, k2 = k, k+1
                i1, i2 = ii[k1], ii[k2]
                if i1 > i2:
                    i1, i2 = i2, i1
                    k1, k2 = k2, k1
                dot[i1, i2] += rr[k1] * rr[k2]
                sql[i1, i2] += rr[k1]**2
                sqr[i1, i2] += rr[k2]**2
                coo[i1, i2] += 1
            self.progress(cur, user_num, 50)

        #dok_matrix不适合进行矩阵算术操作，转为csc格式
        dot = dot.tocsc()
        sql = sql.tocsc()
        sqr = sqr.tocsc()
        coo = coo.tocsc()

        #交互数低于限制全部清零
        dot.data[coo.data < self.min] = 0

        #左右向量平方和的乘积
        sql.data *= sqr.data

        #只需要考虑非0点积
        row, col = dot.nonzero()

        #cosine相似矩阵
        sim[row, col] = dot[row, col] / np.sqrt((sql)[row, col])
        sim[col, row] = sim[row, col]

        return sim.A

    def _train(self):
        if self.use_baseline:
            self.baseline = Baseline()
            self.baseline.train(self.train_dataset)

        user_num = self.train_dataset.matrix.shape[0]
        item_num = self.train_dataset.matrix.shape[1]
        self.sim = self.compute_cosine_similarity(user_num, item_num, self.train_dataset.get_users())
        self.item_means = self.train_dataset.get_item_means()
        self.user_means = self.train_dataset.get_user_means()

    def predict(self, u, i):
        ll, rr = self.train_dataset.get_user(u)
        neighbors = [(sim_i, self.sim[i, sim_i], sim_r) for sim_i, sim_r in zip(ll, rr)]

        neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)[0:self.topk]
        est = self.baseline.predict(u, i) if self.use_baseline else self.item_means[i]
        sum = 0
        divisor = 0

        for sim_i, sim, sim_r in neighbors:
            if not self.use_baseline:
                bias = sim_r - self.item_means[sim_i]
            else:
                bias = sim_r - self.baseline.predict(u, sim_i)

            sum += sim * bias
            divisor += sim

        if divisor != 0:
            est += sum / divisor
        return est

item = Itemcf()
item.compute_cosine_similarity(6040,3671)