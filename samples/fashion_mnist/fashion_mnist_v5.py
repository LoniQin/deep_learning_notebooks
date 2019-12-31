from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
import utils
batch_size = 256
num_inputs = 784
num_outputs = 10
num_hiddens = 256
num_epochs = 20
learning_rate = 0.5
train_iter, test_iter = utils.load_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(num_hiddens, activation='relu'))
net.add(nn.Dense(num_outputs))
net.initialize(init.Normal(sigma=0.01))

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
utils.train_mnist(net = net,
                  train_iter = train_iter,
                  test_iter = test_iter,
                  loss = loss,
                  num_epochs = num_epochs,
                  batch_size = batch_size,
                  params=None,
                  learning_rate=None,
                  trainer= trainer)