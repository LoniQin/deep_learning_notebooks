from mxnet import nd
from mxnet.gluon import nn
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = self.params.get_constant('rand_weight', nd.random.uniform(shape=(20, 20)))
        print(self.rand_weight)
        self.dense = nn.Dense(20, activation='relu')
    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        x = self.dense(x)
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()

if __name__ == "__main__":
    X = nd.random.uniform(shape=(2, 20))
    print(X)
    net = FancyMLP()
    net.initialize()
    print(net.rand_weight)
    print(net(X))