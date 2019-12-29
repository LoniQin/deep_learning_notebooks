from mxnet import autograd, nd
from utils import plot

x = nd.arange(-8, 8, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()
    y.backward()
plot(x, y, 'relu')
plot(x, x.grad, 'grad of relu')

with autograd.record():
    y = x.sigmoid()
    y.backward()
plot(x, y, 'sigmoid')
plot(x, x.grad, 'grad of sigmoid')

with autograd.record():
    y = x.tanh()
    y.backward()
plot(x, y, 'tanh')
plot(x, x.grad, 'grad of tanh')