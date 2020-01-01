from mxnet import init, gluon
from mxnet.gluon import loss as gloss, nn
import utils
from layers.Sequential import Sequential
batch_size = 256
num_inputs = 784
num_outputs = 10
num_hiddens = 256
num_epochs = 100
dropout_prob1 = 0.2
dropout_prob2 = 0.5
learning_rate = 0.1
train_iter, test_iter = utils.load_fashion_mnist(batch_size)
net = Sequential()
net.add(nn.Dense(num_hiddens, activation='relu'))
net.add(nn.Dropout(dropout_prob1))
net.add(nn.Dense(num_hiddens, activation='relu'))
net.add(nn.Dropout(dropout_prob2))
net.add(nn.Dense(num_outputs))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
utils.train_mnist(net=net,
                  train_iter=train_iter,
                  test_iter=test_iter,
                  loss=loss,
                  num_epochs=num_epochs,
                  batch_size=batch_size,
                  trainer=trainer)