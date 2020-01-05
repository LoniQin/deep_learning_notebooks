from mxnet import nd
from mxnet.gluon import nn
from layers.Residual import Residual

def resnet_block(num_channels, num_residuals, first_block = False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk

class ResNet(nn.Sequential):
    def __init__(self, num_outputs, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3))
        self.add(nn.BatchNorm())
        self.add(nn.Activation('relu'))
        self.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        size = 64
        while size <= 512:
            self.add(resnet_block(size, 2, first_block = (size == 64)))
            size *= 2
        self.add(nn.GlobalAvgPool2D())
        self.add(nn.Dense(num_outputs))


if __name__ == "__main__":
    X = nd.random.uniform(shape=(1, 1, 224, 224))
    net = ResNet(num_outputs=10)
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'ouput shape:', X.shape)
