from mxnet import nd
from mxnet.gluon import nn
from layers.FancyMLP import FancyMLP
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'))
        self.net.add(nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')
    def forward(self, X):
        return self.dense(self.net(X))


if __name__ == "__main__":
    X = nd.random.uniform(shape=(784, 10))
    net = nn.Sequential()
    net.add(NestMLP(), nn.Dense(20), FancyMLP())
    net.initialize()
    print(net(X))