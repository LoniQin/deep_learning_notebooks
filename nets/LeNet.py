from mxnet import nd
from mxnet.gluon import nn
class LeNet(nn.Sequential):
    def __init__(self, num_outputs, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        self.add(nn.Conv2D(channels=6, kernel_size=5))
        self.add(nn.BatchNorm())
        self.add(nn.Activation('sigmoid'))
        self.add(nn.MaxPool2D(pool_size=2, strides=2))
        self.add(nn.Conv2D(channels=16, kernel_size=5))
        self.add(nn.BatchNorm())
        self.add(nn.Activation('sigmoid'))
        self.add(nn.MaxPool2D(pool_size=2, strides=2))
        self.add(nn.Dense(120))
        self.add(nn.BatchNorm())
        self.add(nn.Activation('sigmoid'))
        self.add(nn.Dense(84))
        self.add(nn.BatchNorm())
        self.add(nn.Activation('sigmoid'))
        self.add(nn.Dense(num_outputs))