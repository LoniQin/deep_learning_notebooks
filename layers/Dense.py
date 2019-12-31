from mxnet import nd
from mxnet.gluon import nn
class Dense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape = (in_units, units))
        self.bias = self.params.get('bias', shape = (units, ))
    def forward(self, x):
        return nd.relu(nd.dot(x, self.weight.data()) + self.bias.data())
if __name__ == "__main__":
    dense = Dense(units=3, in_units=5)
    print(dense.params)
    dense.initialize()
    print(dense(nd.array([[1, 2, 3, 4, 5]])))

    net = nn.Sequential()
    net.add(Dense(8, in_units=64))
    net.add(Dense(1, in_units=8))
    net.initialize()
    print(net(nd.random.uniform(shape=(2, 64))))