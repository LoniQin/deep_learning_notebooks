from mxnet import init, nd
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
X = nd.random.uniform(shape=(2, 20))
Y = net(X)
print(Y)
print((net[0].params)