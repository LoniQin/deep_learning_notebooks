from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import utils
n_train = 100
learning_rate = 0.01
num_epochs = 100
n_test = 100
true_w = [1.2, -3.4, 5.6]
true_b = 3.0
features = nd.random.normal(scale=1, shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))
labels = nd.sum(nd.array(true_w) * poly_features, axis=1, keepdims=True) + true_b
labels += nd.random.normal(scale=0.1, shape=labels.shape)

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals = None, y2_vals = None, legend = None, figsize = (3.5, 2.5)):
    utils.set_figsize(figsize)
    utils.plt.xlabel(x_label)
    utils.plt.ylabel(y_label)
    utils.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        utils.plt.semilogy(x2_vals, y2_vals, linestyle=":")
        utils.plt.legend(legend)
        utils.plt.show()

def fit_and_plot(train_features, test_features, train_labels, test_labels, loss, num_epochs, learning_rate):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
        print('weight:', net[0].weight.data().asnumpy(), "\nbias:", net[0].bias.data().asnumpy())
    print('Final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net[0].weight.data().asnumpy(), "\nbias:", net[0].bias.data().asnumpy())


fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], gloss.L2Loss(), num_epochs, learning_rate)