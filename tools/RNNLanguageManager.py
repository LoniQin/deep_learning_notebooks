from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import rnn
from nets.RNNNet import RNNNet
import random
import os
import time
import math

def data_iter(indices, batch_size, num_steps, is_random = False, ctx = None):
    num_examples = (len(indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    if is_random:
        random.shuffle(example_indices)
    for i in range(epoch_size):
        batch_indices = example_indices[i * batch_size: (i + 1) * batch_size]
        X = [indices[j * num_steps: (j + 1) * num_steps] for j in batch_indices]
        Y = [indices[j * num_steps + 1: (j + 1) * num_steps + 1] for j in batch_indices]
        yield X, Y

def grad_clipping(params, theta, context):
    norm = nd.array([0], context)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class RNNLanguageParameter:
    def __init__(self,
                 num_steps,
                 indices,
                 index_to_char,
                 char_to_index,
                 filename = None,
                 per_epoch_finish_handler=None,
                 finish_handler=None,
                 num_epochs = 10,
                 batch_size = 32,
                 rnnType = rnn.RNN,
                 num_hiddens = 256,
                 learning_rate = 1e2,
                 clipping_theta = 1e-2,
                 context = None):
        self.num_steps = num_steps
        self.indices = indices
        self.index_to_char = index_to_char
        self.char_to_index = char_to_index
        self.vocabulary_size = len(self.index_to_char)
        self.filename = filename
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.clipping_theta = clipping_theta
        self.num_hiddens = num_hiddens
        self.batch_size = batch_size
        self.context = context
        self.rnnType = rnnType
        self.per_epoch_finish_handler = per_epoch_finish_handler
        self.finish_handler = finish_handler

class RNNLanguageManager:
    def __init__(self, parameter: RNNLanguageParameter):
        self.parameter = parameter

    def initialize(self):
        self.net = RNNNet(self.parameter.num_hiddens, self.parameter.vocabulary_size, rnnType=self.parameter.rnnType)
        if self.parameter.filename != None and os.path.exists(self.parameter.filename):
            self.net.load_parameters(self.parameter.filename, ctx=self.parameter.context)
        else:
            self.net.initialize(force_reinit=True, ctx=self.parameter.context, init=init.Normal(0.01))

    def train(self):
        loss = gloss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.net.collect_params(),
                                'sgd',
                                {
                                    'learning_rate': self.parameter.learning_rate,
                                    'momentum': 0,
                                    'wd': 0
                                })
        for epoch in range(self.parameter.num_epochs):
            l_sum, n, start = 0.0, 0, time.time()
            state = self.net.begin_state(batch_size=self.parameter.batch_size, ctx=self.parameter.context)
            for X, Y in data_iter(self.parameter.indices, self.parameter.batch_size, self.parameter.num_steps, is_random=False, ctx=self.parameter.context):
                for s in state:
                    s.detach()
                with autograd.record():
                    (output, state) = self.net(X, state)
                    y = nd.array(Y).T.reshape((-1,))
                    l = loss(output, y).mean()
                l.backward()
                params = [p.data() for p in self.net.collect_params().values()]
                grad_clipping(params, self.parameter.clipping_theta, self.parameter.context)
                trainer.step(1)
                l_sum += l.asscalar() * y.size
                n += y.size
            epoch_index = epoch + 1
            perplexity = math.exp(l_sum / n)
            time_spent = time.time() - start
            if self.parameter.per_epoch_finish_handler != None:
                self.parameter.per_epoch_finish_handler({
                    'epoch_index': epoch_index,
                    'perplexity': perplexity,
                    'time_spent': time_spent
                })
        if self.parameter.filename != None:
            self.net.save_parameters(self.parameter.filename)
        if self.parameter.finish_handler != None:
            self.parameter.finish_handler()


    def predict(self, prefix, predict_length):
        state = self.net.begin_state(batch_size=1, ctx=self.parameter.context)
        output = [self.parameter.char_to_index[prefix[0]]]
        for t in range(predict_length + len(prefix) - 1):
            X = nd.array([output[-1]], ctx=self.parameter.context).reshape(1, 1)
            (Y, state) = self.net(X, state)
            if t < len(prefix) - 1:
                output.append(self.parameter.char_to_index[prefix[t + 1]])
            else:
                output.append(int(Y.argmax(axis=1).asscalar()))
        return "".join([self.parameter.index_to_char[i] for i in output])