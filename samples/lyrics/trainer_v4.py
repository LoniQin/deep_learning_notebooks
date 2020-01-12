from mxnet import gluon, init, autograd
from mxnet.gluon import rnn
from mxnet.gluon import loss as gloss
import utils
from nets.RNNNet import RNNNet
from samples.lyrics.common import *
import os
import time
import math
def predict(prefix, num_chars, model, context, index_to_char, char_to_index):
    state = model.begin_state(batch_size=1, ctx = context)
    output = [char_to_index[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=context).reshape(1, 1)
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_index[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return "".join([index_to_char[i] for i in output])
def train(net,
          context,
          indices,
          index_to_char,
          char_to_index,
          num_epochs,
          num_steps,
          learning_rate,
          clipping_theta,
          batch_size,
          predict_period,
          predict_length,
          prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd',
                            {
                                'learning_rate': learning_rate,
                                'momentum': 0,
                                'wd': 0
                            })
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        state = net.begin_state(batch_size = batch_size, ctx = context)
        for X, Y in data_iter(indices, batch_size, num_steps, is_random=False, ctx=context):
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = net(X, state)
                y = nd.array(Y).T.reshape((-1, ))
                l = loss(output, y).mean()
            l.backward()
            params = [p.data() for p in net.collect_params().values()]
            grad_clipping(params, clipping_theta, context)
            trainer.step(1)
            l_sum += l.asscalar() * y.size
            n += y.size
        if (epoch + 1) % predict_period == 0:
            print('epoch %d perplexity %f, time %.2f sec' % (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print('-', predict(prefix, predict_length, net, context, index_to_char, char_to_index))

if __name__ == "__main__":
    filename = "trainer_v3.params"
    num_epochs = 10
    learning_rate = 1e2
    clipping_theta = 1e-2
    predict_period = 1
    predict_length = 50
    num_hiddens = 256
    batch_size = 32
    num_steps = 35
    indices, index_to_char, char_to_index, vocabulary_size = read_lyrics()
    context =utils.try_gpu()
    net = RNNNet(num_hiddens, vocabulary_size, rnnType=rnn.LSTM)
    if os.path.exists(filename):
        net.load_parameters(filename, ctx=context)
    else:
        net.initialize(force_reinit=True, ctx=context, init=init.Normal(0.01))
    prefixes = ['分开', '不分开']
    train(net,
          context,
          indices,
          index_to_char,
          char_to_index,
          num_epochs,
          num_steps,
          learning_rate,
          clipping_theta,
          batch_size,
          predict_period,
          predict_length,
          prefixes)
    net.save_parameters(filename)