# -*- coding:utf-8 -*-

from __future__ import division, print_function

import time
import numpy as np


def print_pretty(measures, eval_results):
    """
    格式化输出
    """

    pad = '{:<9}' * (len(measures) + 1)

    print(pad.format('', *measures))

    keep = lambda eval_result:['{:.4f}'.format(single_eval) \
                               for single_eval in eval_result]
    for i, eval_result in enumerate(eval_results):
        print(pad.format('fold {}'.format(i), *keep(eval_result)))
    print(pad.format('avg', *keep(np.mean(eval_results, axis=0))))


class Timer(object):
    """
    time util
    """
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start