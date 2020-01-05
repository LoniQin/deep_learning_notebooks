from mxnet import gluon, init, nd
from mxnet.gluon import nn
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv = False, strides = 1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(X + Y)

if __name__ == "__main__":
    blk = Residual(3)
    blk.initialize()
    X = nd.random.uniform(shape=(4, 3, 6, 6))
    X = blk(X)
    print("shape:", X.shape)

    blk = Residual(3, use_1x1conv=True, strides=3)
    blk.initialize()
    X = nd.random.uniform(shape=(4, 3, 6, 6))
    X = blk(X)
    print("shape:", X.shape)