from mxnet import nd
from mxnet.gluon import loss as gloss
import utils
batch_size = 256
num_inputs = 784
num_outputs = 10
num_hiddens = 256
num_epochs = 20
learning_rate = 0.5
train_iter, test_iter = utils.load_fashion_mnist(batch_size)
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]
for param in params:
    param.attach_grad()

def relu(X):
    return nd.maximum(X, 0)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2

loss = gloss.SoftmaxCrossEntropyLoss()
utils.train_mnist(net = net,
                  train_iter=train_iter,
                  test_iter=test_iter,
                  loss=loss,
                  num_epochs=num_epochs,
                  batch_size=batch_size,
                  params=params,
                  learning_rate=learning_rate)