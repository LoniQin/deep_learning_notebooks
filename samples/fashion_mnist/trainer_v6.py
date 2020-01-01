from mxnet import nd, autograd
from mxnet.gluon import loss as gloss
import utils
batch_size = 256
num_inputs = 784
num_outputs = 10
num_hiddens = 256
num_epochs = 20
dropout_prob1 = 0.2
dropout_prob2 = 0.5
learning_rate = 0.5
train_iter, test_iter = utils.load_fashion_mnist(batch_size)
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_hiddens))
b2 = nd.zeros(num_hiddens)
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b3 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

def dropout(X, dropout_prob):
    assert 0 <= dropout_prob <= 1
    keep_prob = 1 - dropout_prob
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob

def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training():
        H1 = dropout(H1, dropout_prob1)
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():
        H2 = dropout(H2, dropout_prob2)
    return nd.dot(H2, W3) + b3

loss = gloss.SoftmaxCrossEntropyLoss()
utils.train_mnist(net=net,
                  train_iter=train_iter,
                  test_iter=test_iter,
                  loss=loss,
                  num_epochs=num_epochs,
                  batch_size=batch_size,
                  params=params,
                  learning_rate=learning_rate)