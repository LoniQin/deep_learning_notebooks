from mxnet import nd, autograd
from mxnet.gluon import nn
def conv_2d(X, K):
    H, W = X.shape
    h, w = K.shape
    Y = nd.zeros((H - h + 1, W - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1, ))
    def forward(self, X):
        return conv_2d(X, self.weight.data()) + self.bias.data()

if __name__ == "__main__":
    X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    K = nd.array([[1, 2], [3, 4]])
    print(conv_2d(X, K))
    layer = Conv2D(kernel_size=(2, 2))
    layer.initialize()
    layer.weight.set_data(K)
    layer.bias.set_data(nd.array([0]))
    print(layer(X))
    X = nd.ones((6, 8))
    X[:, 2:6] = 0
    print(X)
    K = nd.array([[1, -1]])
    Y = conv_2d(X, K)
    print(Y)
    conv2d = nn.Conv2D(1, kernel_size=(1, 2))
    conv2d.initialize()
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    for i in range(10):
        with autograd.record():
            Y_hat = conv2d(X)
            l = (Y_hat - Y) ** 2
        l.backward()
        conv2d.weight.set_data(conv2d.weight.data() - 0.03 * conv2d.weight.grad())
        print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))
    print(conv2d.weight.data().reshape(shape=(1, 2)))