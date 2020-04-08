from pyfm import pylibfm


from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
train = [
	{"user": "1", "item": "5", "age": 19},
	{"user": "2", "item": "43", "age": 33},
	{"user": "3", "item": "20", "age": 55},
	{"user": "4", "item": "10", "age": 20},
]
v = DictVectorizer()

X = v.fit_transform(train)
print(v.get_feature_names())

print(X.toarray())
y = np.repeat(1.0,X.shape[0])
fm = pylibfm.FM()
fm.fit(X,y)
a = fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
print(a)


def FM_function(dataMatrix, classLabels, k, iter):
    m, n = shape(dataMatrix)
    alpha = 0.01
    w = zeros((n, 1))
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))
    for it in xrange(iter):
        print it
        for x in xrange(m):
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
            p = w_0 + dataMatrix[x] * w + interaction  #
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
            w_0 = w_0 - alpha * loss * classLabels[x]
            for i in xrange(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in xrange(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (
                        dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

    return w_0, w, v
