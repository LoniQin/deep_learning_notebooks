from mxnet import nd
from mxnet.gluon import nn
class Sequential(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)

    def add(self, block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x


if __name__ == "__main__":
    X = nd.random.uniform(shape=(2, 20))
    net = Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    net.initialize()
    print(net(X))