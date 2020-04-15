from __future__ import division, print_function

import numpy as np
import Util.tools as tl
import Util.measure as ms


class Estimator(object):
    """Basic Estimator
    """

    def __init__(self):
        pass

    def train(self, train_dataset):
        self.train_dataset = train_dataset

        with tl.Timer() as t:
            self._train()

        print("{} algorithm train process cost {:.3f} sec".
              format(self.__class__.__name__, t.interval))

    def _train(self):
        raise NotImplementedError()

    def predict(self, u, i):
        raise NotImplementedError()

    def estimate(self, raw_test_dataset, measures):
        with tl.Timer() as t:
            error = self._estimate(raw_test_dataset, measures)

        print("{} algorithm predict process cost {:.3f} sec".
              format(self.__class__.__name__, t.interval))
        return error

    def _estimate(self, raw_test_dataset, measures):
        users_mean = self.train_dataset.get_user_means()
        items_mean = self.train_dataset.get_item_means()

        all = len(raw_test_dataset)
        errors = []
        cur = 0
        alg_count = 0

        for raw_u, raw_i, r, _ in raw_test_dataset:
            cur += 1
            has_raw_u = raw_u in self.train_dataset.uid_dict
            has_raw_i = raw_i in self.train_dataset.iid_dict

            if not has_raw_u and not has_raw_i:
                real, est = r, self.train_dataset.global_mean
            elif not has_raw_u:
                i = self.train_dataset.iid_dict[raw_i]
                real, est = r, items_mean[i]
            elif not has_raw_i:
                u = self.train_dataset.uid_dict[raw_u]
                real, est = r, users_mean[u]
            else:
                u = self.train_dataset.uid_dict[raw_u]
                i = self.train_dataset.iid_dict[raw_i]
                real, est = r, self.predict(u, i)
                alg_count += 1

            est = min(5, est)
            est = max(1, est)
            errors.append(real - est)

            self.progress(cur, all, 2000)

        fold_eval_result = [getattr(ms, measure)(errors) for measure in measures]
        return fold_eval_result

    @staticmethod
    def progress(cur, all, bin=50):
        if cur % bin == 0 or cur == all:
            progress = 100 * (cur / all)
            print("progress: {:.2f}%".format(progress))


class IterationEstimator(Estimator):
    """Iterator Estimator
    """

    def _train(self):
        self._prepare()
        for current_epoch in range(self.n_epochs):
            print(" processing epoch {}".format(current_epoch))
            self._iteration()
            print(" cur train rmse {}".format(self._eval()))

    def _prepare(self):
        """
        do some prepare work
        """

        raise NotImplementedError()

    def _iteration(self):
        """
        core iteration
        """

        raise NotImplementedError()

    def _pred(self):
        """
        core pred process
        """

        raise NotImplementedError()

    def _eval(self):
        """
        eval on valid dateset
        """

        pred_ratings = self._pred()
        real_ratings = self.train_dataset.matrix
        idx = real_ratings.nonzero()
        bias = np.asarray(pred_ratings[idx] - real_ratings[idx])
        return np.sqrt(np.sum(bias ** 2) / real_ratings.count_nonzero())