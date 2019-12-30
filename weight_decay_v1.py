from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import utils

def l2_penalty(w):
    return (w ** 2).sum()/ 2

n_train = 20
n_test = 100
num_inputs = 200
true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05
batch_size = 1
num_epochs = 100
learning_rate = 0.003
features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
train_features, test_features = features[:n_train], features[n_train:]
train_labels, test_labels = labels[:n_train], labels[n_train:]
net = utils.lineareg
loss = utils.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
def fit_and_plot(lambd):
    w = nd.random.normal(scale=1, shape=true_w.shape)
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            utils.sgd([w, b], learning_rate, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().asscalar())
    utils.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print("L2 norm of w:", w.norm().asscalar())

fit_and_plot(0)
fit_and_plot(3)