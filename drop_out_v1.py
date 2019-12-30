from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
def dropout(X, dropout_prob):
    assert 0 <= dropout_prob <= 1
    keep_prob = 1 - dropout_prob
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob

X = nd.arange(16).reshape((2, 8))
print(X)
print(dropout(X, 0.5))
print(dropout(X, 0))
print(dropout(X, 1))