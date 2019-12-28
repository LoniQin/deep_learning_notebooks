from mxnet import autograd, nd, init, gluon
from mxnet.gluon import data as gdata, nn, loss as gloss
num_inputs = 2
num_examples = 1000
batch_size = 10
num_epochs = 10
true_w = nd.array([2, -3.4])
true_b = nd.array([4.2])
# Generate random features
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
# Calcuate the labels and add noise
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.1, shape=labels.shape)
# Create dataset using input and output we create
dataset = gdata.ArrayDataset(features, labels)
# This data loader shuffle datas
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
# Build neural network with just 1 dense layer
net = nn.Sequential()
# Dense layer with just 1 feature output
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
# Loss function
loss = gloss.L2Loss()
# Trainer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
# Train many epochs
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('Epoch:%d, loss: %f' % (epoch, l.mean().asnumpy()))
    # Compare predicted weight and bias the real weight and bias
    dense = net[0]
    print(true_w, dense.weight.data())
    print(true_b, dense.bias.data())