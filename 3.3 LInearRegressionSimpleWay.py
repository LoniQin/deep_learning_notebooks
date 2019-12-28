from mxnet import autograd, nd
from base import lineareg, squared_loss, sgd
from mxnet.gluon import data as gdata, nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon
num_inputs = 2
num_examples = 1000
true_w = nd.array([2, -3.4])
true_b = nd.array([4.2])
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
w = nd.random.normal(scale=1, shape=true_w.shape)
b = nd.random.normal(scale=1, shape=true_b.shape)
labels = lineareg(features, true_w, true_b)
labels += nd.random.normal(scale=0.1, shape=labels.shape)
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('Epoch:%d, loss: %f' % (epoch, l.mean().asnumpy()))
    print("params:", net.collect_params())