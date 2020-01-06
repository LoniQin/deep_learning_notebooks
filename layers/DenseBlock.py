from mxnet import gluon, init, nd
from mxnet.gluon import nn
def conv_block(num_channels):
    block = nn.Sequential()
    block.add(nn.BatchNorm())
    block.add(nn.Activation('relu'))
    block.add(nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return block

class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for block in self.net:
            Y = block(X)
            X = nd.concat(X, Y, dim = 1)
        return X

if __name__ == "__main__":
    block = DenseBlock(2, 10)
    block.initialize()
    X = nd.random.uniform(shape=(4, 3, 8, 8))
    Y = block(X)
    print(Y.shape)