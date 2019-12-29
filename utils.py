import random
from IPython import display
import matplotlib.pyplot as plt
from mxnet import nd, autograd
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

def evaluate_accuracy(data_iter, net):
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