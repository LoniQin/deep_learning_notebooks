import random
from IPython import display
import matplotlib.pyplot as plt
from mxnet import nd, autograd
from mxnet.gluon import data as gdata
import mxnet
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)

def lineareg(X, w, b):
    return nd.dot(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y) ** 2 / 2

def sgd(params, lr, batch_size):
    for i in range(len(params)):
        nd.elemwise_sub(params[i], lr * params[i].grad / batch_size, out=params[i])


def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def get_fashion_mnist_labels(labels):
    text_Labels =['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_Labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize = (12, 12))
    for fig, image, label in zip(figs, images, labels):
        fig.imshow(image.reshape((28, 28)).asnumpy())
        fig.set_title(label)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

def evaluate_accuracy(data_iter, net, ctx = None):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n

def cross_entropy(y_hat, y):
    return  -nd.pick(y_hat, y).log()

def softmax(X):
    x_exp = X.exp()
    partition = x_exp.sum(axis = 1, keepdims = True)
    return x_exp / partition

def train_mnist(net, train_iter, test_iter, loss, num_epochs, batch_size, params = None, learning_rate = None, trainer = None):
    for epoch in range(num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                sgd(params, learning_rate, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype("float32")
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print("Epoch:%d loss: %.4f train acc %.3f test acc %.3f" % (epoch, train_l_sum / n, train_acc_sum / n, test_acc))

def plot(x_vals, y_vals, name):
    set_figsize((5, 2.5))
    plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.show()


def load_fashion_mnist(batch_size):
    mnist_train = gdata.vision.FashionMNIST(train=True)

    mnist_test = gdata.vision.FashionMNIST(train=False)

    transformer = gdata.vision.transforms.ToTensor()

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True,
                                  num_workers=4)

    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False,
                                 num_workers=4)
    return train_iter, test_iter

def load_fashion_mnist_v2(batch_size,resize=None):
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)

    mnist_train = gdata.vision.FashionMNIST(train=True)

    mnist_test = gdata.vision.FashionMNIST(train=False)

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True,
                                  num_workers=4)

    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False,
                                 num_workers=4)
    return train_iter, test_iter


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals = None, y2_vals = None, legend = None, figsize = (3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=":")
        plt.legend(legend)
        plt.show()

def dropout(X, dropout_prob):
    assert 0 <= dropout_prob <= 1
    keep_prob = 1 - dropout_prob
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob

def conv_2d(X, K):
    H, W = X.shape
    h, w = K.shape
    Y = nd.zeros((H - h + 1, W - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def pool_2d(X, pool_size, mode = 'max'):
    Y = nd.zeros((X.shape[0] - pool_size + 1, X.shape[1] - pool_size + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+pool_size, j: j+pool_size].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + pool_size, j: j + pool_size].mean()


def try_gpu():
    try:
        ctx = mxnet.gpu()
        _ = nd.zeros((1, ), ctx=ctx)
    except mxnet.base.MXNetError:
        ctx = mxnet.cpu()
    return ctx