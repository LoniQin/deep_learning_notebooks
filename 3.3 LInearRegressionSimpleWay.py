from mxnet import autograd, nd, init, gluon
from mxnet.gluon import data as gdata, nn, loss as gloss
num_inputs = 2
num_examples = 1000
batch_size = 10
num_epochs = 10
true_w = nd.array([2, -3.4])
true_b = nd.array([4.2])
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
w = nd.random.normal(scale=1, shape=true_w.shape)
b = nd.random.normal(scale=1, shape=true_b.shape)
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.1, shape=labels.shape)
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('Epoch:%d, loss: %f' % (epoch, l.mean().asnumpy()))
    dense = net[0]
    print(true_w, dense.weight.data())
    print(true_b, dense.bias.data())