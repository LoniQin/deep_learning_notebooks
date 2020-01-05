from mxnet import init, gluon, autograd, nd
from mxnet.gluon import loss as gloss, nn
from layers.Inception import Inception
import utils
import time
import os

file_name = 'mnist_v13.params'
batch_size = 128
num_outputs = 10
num_epochs = 1
learning_rate = 0.1
dropout_rate = 0.5
data_length = 60000.0
train_iter, test_iter = utils.load_fashion_mnist_v2(batch_size, 96)
ctx = utils.try_gpu()
# b1
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'))
# b2
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'))
b2.add(nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'))
b2.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
# b3
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32))
b3.add(Inception(128, (128, 194), (32, 96), 64))
b3.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
# b4
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64))
b4.add(Inception(160, (112, 224), (24, 64), 64))
b4.add(Inception(128, (128, 256), (24, 64), 64))
b4.add(Inception(112, (144, 288), (32, 64), 64))
b4.add(Inception(256, (160, 320), (32, 128), 128))
b4.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

# b5
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128))
b5.add(Inception(384, (192, 384), (48, 128), 128))
b5.add(nn.GlobalAvgPool2D())
net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(num_outputs))
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
        print("time: %.2fs, progress:%.2f, estimated time: %.2fs"%(time.time() - start, n / data_length, (time.time() - start) * data_length / n))
    test_acc = utils.evaluate_accuracy(test_iter, net)
    print("Epoch:%d loss: %.4f train acc %.3f test acc %.3f time:%.2fs" % (epoch, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))
    net.save_parameters(file_name)