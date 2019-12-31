from mxnet import nd
from mxnet.gluon import nn
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)
    def forward(self, x):
        return self.output(self.hidden(x))


if __name__ == "__main__":
    filename = 'mlp.params'
    X = nd.random.uniform(shape=(2, 20))
    net = MLP()
    net.initialize()
    net(X)
    net.save_parameters(filename)
    net2 = MLP()
    net2.load_parameters(filename)
    print(net(X) == net2(X))