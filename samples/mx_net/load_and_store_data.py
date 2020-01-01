from mxnet import nd
from mxnet.gluon import nn
x = nd.ones(3)
print(x)
nd.save('x', x)
x2 = nd.load('x')
print(x2)
y = nd.zeros(4)
nd.save('xy', [x, y])
x2, y2 = nd.load('xy')
print(x2)
print(y2)