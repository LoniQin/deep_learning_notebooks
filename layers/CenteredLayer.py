from mxnet import gluon, nd
from mxnet.gluon import nn
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return  x - x.mean()
if __name__ == "__main__":
    layer = CenteredLayer()
    print(layer(nd.array([1, 2, 3, 4, 5])))
    net = nn.Sequential()
    net.add(nn.Dense(128), CenteredLayer())
    net.initialize()
    print(net(nd.uniform(shape=(4, 10))).mean().asscalar())
    params = gluon.ParameterDict()
    params.get('param2', shape = (2, 8))
    print(params)