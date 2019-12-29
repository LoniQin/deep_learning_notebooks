from mxnet.gluon import data as gdata
import time
from mxnet import autograd, nd

num_inputs = 784
num_outputs = 10
batch_size = 256
num_workers = 4
num_epochs = 100
learning_rate = 0.1
mnist_train = gdata.vision.FashionMNIST(train=True)

mnist_test = gdata.vision.FashionMNIST(train=False)

transformer = gdata.vision.transforms.ToTensor()

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=num_workers)

test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False, num_workers=num_workers)

def softmax(X):
    x_exp = X.exp()
    partition = x_exp.sum(axis = 1, keepdims = True)
    return x_exp / partition

def net(X, w, b):
    y = nd.dot(X.reshape((-1, w.shape[0])), w) + b
    return softmax(y)

def cross_entropy(y_hat, y):
    return  -nd.pick(y_hat, y).log()

def sgd(params, lr, batch_size):
    for i in range(len(params)):
        nd.elemwise_sub(params[i], lr * params[i].grad / batch_size, out=params[i])

w = nd.random.normal(scale=1.0, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
w.attach_grad()
b.attach_grad()
loss = cross_entropy
# Train models
for epoch in range(1, num_epochs + 1):
    start = time.time()
    acc_sum, n = 0.0, 0
    total = float(len(mnist_train))
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X, w, b)
            l = loss(y_hat, y).sum()
        l.backward()
        sgd([w, b], learning_rate, batch_size)
        acc_sum += (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()
    print("Epoch:%d Elapsed time:%.2f accuracy:%.2f%%" % (epoch, time.time() - start, (acc_sum / total) * 100))