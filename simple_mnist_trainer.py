import utils
from mxnet import nd
from mxnet.gluon import data as gdata
def net(X):
    return utils.softmax(nd.dot(X.reshape((-1, num_inputs)), w) + b)
num_inputs = 784
num_outputs = 10
batch_size = 256
num_workers = 4
num_epochs = 100
learning_rate = 0.1
w = nd.random.normal(scale=1.0, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
w.attach_grad()
b.attach_grad()
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
transformer = gdata.vision.transforms.ToTensor()
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False, num_workers=num_workers)
utils.train_mnist(net, train_iter, test_iter, utils.cross_entropy, num_epochs, batch_size, [w, b], learning_rate)