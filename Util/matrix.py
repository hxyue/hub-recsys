from __future__ import division, print_function

import numpy as np
import itertools


class Matrix(object):

    def __init__(self, sparse_matrix, uid_dict=None, iid_dict=None):
        self.matrix = sparse_matrix.tocsc()
        self._global_mean = None
        coo_matrix = sparse_matrix.tocoo()
        self.uids = set(coo_matrix.row)
        self.iids = set(coo_matrix.col)
        self.uid_dict = uid_dict
        self.iid_dict = iid_dict

    def get_item(self, i):
        """
        (is, (us, rs))
        """

        ratings = self.matrix.getcol(i).tocoo()
        return ratings.row, ratings.data

    def get_user(self, u):
        """
        (u, (is, rs))
        """

        ratings = self.matrix.getrow(u).tocoo()
        return ratings.col, ratings.data

    def get_users(self):
        """
        iterator(u, (is, rs))
        """
        for u in self.get_uids():
            yield u, self.get_user(u)

    def get_user_means(self):
        """
        用户的平均评分字典
        """

        users_mean = {}
        for u in self.get_uids():
            users_mean[u] = np.mean(self.get_user(u)[1])
        return users_mean

    def  get_item_means(self):
        """
        物品的平均评分字典
        """

        item_means = {}
        for i in self.get_iids():
            item_means[i] = np.mean(self.get_item(i)[1])
        return item_means

    def all_ratings(self, axis=1):
        """
        row(u,i,r)
        or
        col(u, i, r)
        """
        coo_matrix = self.matrix.tocoo()

        if axis == 1:
            return zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
        else:
            return coo_matrix.row, coo_matrix.col, coo_matrix.data

    def get_uids(self):
        """
        所有用户id集
        """

        return np.unique(self.matrix.tocoo().row)

    def get_iids(self):
        """
        所有物品id集
        """
        return np.unique(self.matrix.tocoo().col)

    def has_user(self, u):
        """
        是否存在用户u
        """

        return u in self.uids

    def has_item(self, i):
        """
        是否存在物品i
        """

        return i in self.iids

    @property
    def global_mean(self):
        """
        全局均值
        """

        if self._global_mean is None:
            self._global_mean = np.sum(self.matrix.data) / self.matrix.size
        return self._global_mean