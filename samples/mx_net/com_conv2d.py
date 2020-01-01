from mxnet import nd
from mxnet.gluon import nn
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1) 代表批量大小和通道数为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

if __name__ == "__main__":
    X = nd.random.uniform(shape=(8, 8))
    conv2d = nn.Conv2D(1, kernel_size=3, padding=0)
    print(comp_conv2d(conv2d, X).shape)
    conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
    print(comp_conv2d(conv2d, X).shape)
    conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
    print(comp_conv2d(conv2d, X).shape)
    conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
    print(comp_conv2d(conv2d, X).shape)
    conv2d = nn.Conv2D(1, kernel_size=3, padding=2, strides=2)
    print(comp_conv2d(conv2d, X).shape)
    conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
    print(comp_conv2d(conv2d, X).shape)