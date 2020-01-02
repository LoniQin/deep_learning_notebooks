from mxnet import init, gluon, autograd, nd
from mxnet.gluon import loss as gloss, nn
import utils
import time
import os
def nin_block(num_channels, kernel_size, strides, padding):
    block = nn.Sequential()
    block.add(nn.Conv2D(num_channels, kernel_size, strides, padding, activation='relu'))
    block.add(nn.Conv2D(num_channels, 1, activation='relu'))
    block.add(nn.Conv2D(num_channels, 1, activation='relu'))
    return block

file_name = 'mnist_v12.params'
batch_size = 128
num_outputs = 10
num_epochs = 1
learning_rate = 0.1
dropout_rate = 0.5
train_iter, test_iter = utils.load_fashion_mnist_v2(batch_size, 224)
ctx = utils.try_gpu()
ratio = 4
net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0))
net.add(nn.MaxPool2D(pool_size=3, strides=2))
net.add(nin_block(256, kernel_size=5, strides=1, padding=2))
net.add(nn.MaxPool2D(pool_size=3, strides=2))
net.add(nin_block(384, kernel_size=3, strides=1, padding=1))
net.add(nn.MaxPool2D(pool_size=3, strides=2))
net.add(nn.Dropout(0.5))
net.add(nin_block(10, kernel_size=3, strides=1, padding=1))
net.add(nn.GlobalAvgPool2D())
net.add(nn.Flatten())
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
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
        print("time: %.2fs, progress:%.2f, estimated time: %.2fs"%(time.time() - start, n / 50000.0, (time.time() - start) * 50000.0 / n))
    test_acc = utils.evaluate_accuracy(test_iter, net)
    print("Epoch:%d loss: %.4f train acc %.3f test acc %.3f time:%.2fs" % (epoch, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))
    net.save_parameters(file_name)