from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
from utils import lineareg, squared_loss, data_iter, sgd
row_count = 1000
column_count = 2
X = nd.random.normal(scale=1, shape=(row_count, column_count))
w = nd.array([[2], [-3.4]])
b = 4.2
epsilon = nd.random.normal(scale=0.01, shape=[row_count, 1])
y = nd.dot(X, w) + b + epsilon
y
def use_svg_display():
    display.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
set_figsize()
plt.scatter(X[:, 1].asnumpy(), y.asnumpy(), 1)
plt.show()


lr = 0.03
batch_size = 10
num_epochs = 3
net = lineareg
loss = squared_loss
features = nd.random.normal(1, shape=(row_count, column_count))
true_w = nd.random.normal(scale=0.01, shape=(column_count, 1))
true_b = nd.random.normal(scale=1, shape=(1,))
w = nd.random.normal(scale=0.01, shape=(column_count, 1))
b = nd.random.normal(scale=1, shape=(1,))
labels = net(X, true_w, true_b)
labels += nd.random.normal(scale=0.01, shape=labels.shape)
w.attach_grad()
b.attach_grad()
epsilon = nd.random.normal(0.01, shape=(row_count, 1))
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_l = loss(net(X, w, b), y)
    print("Epoch: %d, loss: %f"%(epoch + 1, train_l.mean().asnumpy()))
    print("W: ", w.asnumpy(), " true W:", true_w.asnumpy())
    print("b: ", b.asnumpy(), " true b:", true_b.asnumpy())