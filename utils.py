import random
from mxnet import nd
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)

def lineareg(X, w, b):
    return nd.dot(X, w) + b
def squared_loss(y_hat, y):
    return (y_hat - y) ** 2 / 2

def sgd(params, lr, batch_size):
    for i in range(len(params)):
        nd.elemwise_sub(params[i], lr * params[i].grad / batch_size, out=params[i])