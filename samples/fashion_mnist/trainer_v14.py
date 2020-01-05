from mxnet import init, gluon, autograd
from mxnet.gluon import loss as gloss, nn
from layers.BatchNorm import BatchNorm
import utils
import time
import os
file_name = 'mnist_v14.params'
batch_size = 256
num_outputs = 10
num_epochs = 20
learning_rate = 0.2
data_length = 60000.0
train_iter, test_iter = utils.load_fashion_mnist(batch_size)
ctx = utils.try_gpu()
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5))
net.add(BatchNorm(6, num_dims=4))
net.add(nn.Activation('sigmoid'))
net.add(nn.MaxPool2D(pool_size=2, strides=2))
net.add(nn.Conv2D(channels=16, kernel_size=5))
net.add(BatchNorm(16, num_dims=4))
net.add(nn.Activation('sigmoid'))
net.add(nn.MaxPool2D(pool_size=2, strides=2))
net.add(nn.Dense(120))
net.add(BatchNorm(120, num_dims=2))
net.add(nn.Activation('sigmoid'))
net.add(nn.Dense(84))
net.add(BatchNorm(84, num_dims=2))
net.add(nn.Activation('sigmoid'))
net.add(nn.Dense(num_outputs))
if os.path.exists(file_name):
    net.load_parameters(file_name)
else:
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
for epoch in range(1, num_epochs + 1):
    train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
    for X, y in train_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y).sum()
        l.backward()
        trainer.step(batch_size)
        y = y.astype("float32")
        train_l_sum += l.asscalar()
        train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
        n += y.size
        print("time: %.2fs, progress:%.2f, estimated time: %.2fs" % ( time.time() - start, n / data_length, (time.time() - start) * data_length / n))
    test_acc = utils.evaluate_accuracy(test_iter, net)
    print("Epoch:%d loss: %.4f train acc %.3f test acc %.3f time:%.2fs" % (epoch, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))
net.save_parameters(file_name)