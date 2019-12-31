from mxnet import nd
from mxnet.gluon import nn
x = nd.ones(3)
print(x)
nd.save('x', x)
x2 = nd.load('x')
print(x2)