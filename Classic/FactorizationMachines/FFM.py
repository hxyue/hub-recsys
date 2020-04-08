import numpy as np
import math

class FM:
    def __init__(self,num_factors=10,
                 num_iter=1,
                 k0=True,
                 k1=True,
                 init_stdev=0.1,
                 validation_size=0.01,
                 learning_rate_schedule="optimal",
                 initial_learning_rate=0.01,
                 power_t=0.5,
                 t0=0.001,
                 task='classification',
                 verbose=True,
                 shuffle_training=True,
                 seed = 28):
        self.w0 = w0
        self.w = w
        self.v = v
        self.num_factors = num_factors
        self.num_attributes = num_attributes
        self.n_iter = n_iter
        self.k0 = k0
        self.k1 = k1
        self.t = 1
        self.t0 = t0
        self.learning_rate = eta0
        self.power_t = power_t
        self.min_target = min_target
        self.max_target = max_target
        self.sum = np.zeros(self.num_factors)
        self.sum_sqr = np.zeros(self.num_factors)
        self.task = task
        self.learning_rate_schedule = learning_rate_schedule
        self.shuffle_training = shuffle_training
        self.seed = seed
        self.verbose = verbose

        self.reg_0 = 0.0
        self.reg_w = 0.0
        self.reg_v = np.zeros(self.num_factors)

        self.sumloss = 0.0
        self.count = 0

        self.grad_w = np.zeros(self.num_attributes)
        self.grad_v = np.zeros((self.num_factors, self.num_attributes))




def _sgd_theta_step(self, x_data_ptr, x_ind_ptr, xnnz, y):
    mult = 0.0
    w0 = self.w0
    w = self.w
    v = self.v
    grad_w = self.grad_w
    grad_v = self.grad_v
    learning_rate = self.learning_rate
    reg_0 = self.reg_0
    reg_w = self.reg_w
    reg_v = self.reg_v

    p = self._predict_instance(x_data_ptr, x_ind_ptr, xnnz)

    if self.task == "regression":
        p = min(self.max_target, p)
        p = max(self.min_target, p)
        mult = 2 * (p - y);
    else:
        mult = y * ((1.0 / (1.0 + math.exp(-y * p))) - 1.0)



    # Update global bias
    if self.k0:
        grad_0 = mult
        w0 -= learning_rate * (grad_0 + 2 * reg_0 * w0)

    # Update feature biases
    if self.k1:
        for i in range(xnnz):
            feature = x_ind_ptr[i]
            grad_w[feature] = mult * x_data_ptr[i]
            w[feature] -= learning_rate * (grad_w[feature]
                                           + 2 * reg_w * w[feature])
    # Update feature factor vectors
    for f in range(self.num_factors):
        for i in range(xnnz):
            feature = x_ind_ptr[i]
            grad_v[f, feature] = mult * (x_data_ptr[i] * (self.sum[f] - v[f, feature] * x_data_ptr[i]))
            v[f, feature] -= learning_rate * (grad_v[f, feature] + 2 * reg_v[f] * v[f, feature])

        # Pass updated vars to other functions
        self.learning_rate = learning_rate
        self.w0 = w0
        self.w = w
        self.v = v
        self.grad_w = grad_w
        self.grad_v = grad_v

        self.t += 1
        self.count += 1

    def _sgd_theta_step(self, x_data_ptr, x_ind_ptr, xnnz, y):

    def _predict_instance(self, x_data_ptr, x_ind_ptr, xnnz):
        result = 0.0
        i = 0
        f = 0

        w0 = self.w0
        w = self.w
        v = self.v
        sum_ = np.zeros(self.num_factors)
        sum_sqr_ = np.zeros(self.num_factors)
        if self.k0:
            result += w0
        if self.k1:
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                result += w[feature] * x_data_ptr[i]
        for f in range(self.num_factors):
            sum_[f] = 0.0
            sum_sqr_[f] = 0.0
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                d = v[f, feature] * x_data_ptr[i]
                sum_[f] += d
                sum_sqr_[f] += d * d
            result += 0.5 * (sum_[f] * sum_[f] - sum_sqr_[f])

        # pass sum to sgd_theta
        self.sum = sum_
        return result

    def fit(self, dataset, validation_dataset):

        n_samples = dataset.n_samples
        n_validation_samples = validation_dataset.n_samples
        y = 0.0
        validation_y = 0.0
        count = 0
        epoch = 0
        i = 0
        sample_weight = 1.0
        validation_sample_weight = 1.0

        for epoch in range(self.n_iter):

            if self.verbose > 0:
                print("-- Epoch %d" % (epoch + 1))
            self.count = 0
            self.sumloss = 0
            if self.shuffle_training:
                dataset.shuffle(self.seed)

            for i in range(n_samples):
                dataset.next( & x_data_ptr, & x_ind_ptr, & xnnz, & y,
                & sample_weight)

                self._sgd_theta_step(x_data_ptr, x_ind_ptr, xnnz, y)

                if epoch > 0:
                    validation_dataset.next( & validation_x_data_ptr, & validation_x_ind_ptr,
                    & validation_xnnz, & validation_y,
                    & validation_sample_weight)
                    self._sgd_lambda_step(validation_x_data_ptr, validation_x_ind_ptr,
                                          validation_xnnz, validation_y)
            if self.verbose > 0:
                error_type = "MSE" if self.task == REGRESSION else "log loss"
                print
                "Training %s: %.5f" % (error_type, (self.sumloss / self.count))

    def _predict(self, dataset):

        n_samples = dataset.n_samples
        return_preds = np.zeros(n_samples)

        for i in range(n_samples):
            dataset.next( & x_data_ptr, & x_ind_ptr, & xnnz, & y_placeholder,
            & sample_weight)
            p = self._predict_instance(x_data_ptr, x_ind_ptr, xnnz)
            if self.task == "regression":
                p = min(self.max_target, p)
                p = max(self.min_target, p)
            else:
                p = (1.0 / (1.0 + math.exp(-p)))
            return_preds[i] = p
        return return_preds



