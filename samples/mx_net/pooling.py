from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
def pool2d(X, pool_size, mode = 'max'):
    Y = nd.zeros((X.shape[0] - pool_size + 1, X.shape[1] - pool_size + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+pool_size, j: j+pool_size].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + pool_size, j: j + pool_size].mean()
    return Y
if __name__ == "__main__":
    X = nd.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
    pool_size = 2
    print("X:", X, " shape:", X.shape)
    Y = nd.zeros(shape=(X.shape[0] - pool_size + 1, X.shape[1] - pool_size + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = X[i: i + pool_size, j:j+pool_size].max()
    print("Max pooling:", Y, " shape:", Y.shape)
    Y = nd.zeros(shape=(X.shape[0] - pool_size + 1, X.shape[1] - pool_size + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = X[i: i + pool_size, j:j + pool_size].mean()
    print("Average pooling:", Y, " shape:", Y.shape)
    Y = pool2d(X, pool_size, 'max')
    print("Max pooling:", Y, " shape:", Y.shape)
    Y = pool2d(X, pool_size, 'avg')
    print("Average pooling:", Y, " shape:", Y.shape)
    X = nd.arange(16).reshape((1, 1, 4, 4))
    print("X:", X, " shape:", X.shape)
    pool_2d = nn.MaxPool2D(3)
    Y = pool_2d(X)
    print("Max pooling:", Y, " shape:", Y.shape)
    pool_2d = nn.AvgPool2D(3)
    Y = pool_2d(X)
    print("Average pooling:", Y, " shape:", Y.shape)
    pool_2d = nn.MaxPool2D(2, padding=1, strides=2)
    Y = pool_2d(X)
    print("Max pooling:", Y, " shape:", Y.shape)
    pool_2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
    Y = pool_2d(X)
    print("Max pooling:", Y, " shape:", Y.shape)

    X = nd.concat(X, X + 1, dim = 1)
    print("X:", X, " shape:", X.shape)
    Y = pool_2d(X)
    print("Max pooling:", Y, " shape:", Y.shape)