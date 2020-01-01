from mxnet import gluon, nd
from mxnet.gluon import nn
from layers.Sequential import Sequential
from layers.Dense import Dense
class Centered(nn.Block):
    def __init__(self, **kwargs):
        super(Centered, self).__init__(**kwargs)
    def forward(self, x):
        return  x - x.mean()
if __name__ == "__main__":
    layer = Centered()
    net = Sequential()
    net.add(Dense(in_units=10, units=128))
    net.add(Centered())
    net.initialize()
    print(net(nd.uniform(shape=(4, 10))).mean().asscalar())
    params = gluon.ParameterDict()
    params.get('param2', shape = (2, 8))
    print(params)