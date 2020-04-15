# -*- coding:utf-8 -*-

from __future__ import division, print_function

import numpy as np


def rmse(errors):
    return np.sqrt(np.mean(np.power(errors, 2)))


def mae(errors):
    return np.mean(np.abs(errors))