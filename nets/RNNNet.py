from mxnet import nd
from mxnet.gluon import nn, rnn

class RNNNet(nn.Block):

    def __init__(self, num_hiddens, vocabulary_size, rnnType = rnn.RNN, **kwargs):
        super(RNNNet, self).__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.rnn = rnnType(num_hiddens)
        self.dense = nn.Dense(vocabulary_size)

    def forward(self, inputs, state):
        X = nd.one_hot(nd.array(inputs).T, self.vocabulary_size)
        Y, state = self.rnn(X, state)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)