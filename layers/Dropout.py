from mxnet import nd
from mxnet.gluon import nn
class Dropout(nn.Block):
    def __init__(self, rate, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        assert 0 <= rate <= 1
        self.rate = rate
    def forward(self, x):
        keep_rate = 1 - self.rate
        if keep_rate == 0: return X.zeros_like()
        mask = nd.random.uniform(0, 1, X.shape) < keep_rate
        return mask * X / keep_rate
if __name__ == "__main__":
    X = nd.arange(16).reshape((2, 8))
    print(X)
    dropout = Dropout(rate=1)
    print(dropout(X))
    dropout = Dropout(rate=0.5)
    print(dropout(X))
    dropout = Dropout(rate=0)
    print(dropout(X))