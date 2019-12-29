import utils
from mxnet import gluon, init
from mxnet.gluon import data as gdata, nn, loss as gloss
num_inputs = 784
num_outputs = 10
batch_size = 256
num_workers = 4
num_epochs = 10
learning_rate = 0.1
# Load data
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
transformer = gdata.vision.transforms.ToTensor()
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False, num_workers=num_workers)
# Construct neural network model
net = nn.Sequential()
net.add(nn.Dense(num_outputs))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
utils.train_mnist(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)