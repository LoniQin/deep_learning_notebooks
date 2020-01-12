from samples.lyrics.common import *


def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]



def get_params():
    # Hidden layer
    W_xh = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens), ctx=context)
    W_hh = nd.random.normal(scale=0.01, shape=(num_hiddens, num_hiddens), ctx=context)
    b_h = nd.zeros(num_hiddens, ctx=context)
    # Output layer
    W_hq = nd.random.normal(scale=0.01, shape=(num_hiddens, num_output), ctx=context)
    b_q = nd.zeros(num_output, ctx=context)
    # Attach gradient
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for item in params:
        item.attach_grad()
    return params

def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

def predict(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocabulary_size, ctx, index_to_char, char_to_index):
    state = init_rnn_state(1, num_hiddens, context)
    output = [char_to_index[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocabulary_size)
        # Calcuate output and update hidden layer state
        (Y, state) = rnn(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_index[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join(index_to_char[i] for i in output)



def train(rnn,
          get_params,
          init_rnn_state,
          num_hiddens,
          vocabulary_size,
          context,
          indices,
          index_to_char,
          char_to_index,
          is_random_iter,
          num_epochs,
          num_steps,
          learning_rate,
          clipping_theta,
          batch_size,
          predict_period,
          predict_length,
          prefixes):
     params = get_params()
     loss = gloss.SoftmaxCrossEntropyLoss()
     for epoch in range(num_epochs):
         l_sum = 0.0
         n = 0
         start = time.time()
         if is_random_iter == False:
             state = init_rnn_state(batch_size, num_hiddens, context)
         items = [char_to_index[c] for c in corpus_indices]
         for X, y in data_iter(items, batch_size, num_steps, is_random_iter, context):
             if is_random_iter:
                 state = init_rnn_state(batch_size, num_hiddens, context)
             else:
                 for s in state:
                     s.detach()
             with autograd.record():
                 inputs = to_onehot(nd.array(X), vocabulary_size)
                 (outputs, state) = rnn(inputs, state, params)
                 outputs = nd.concat(*outputs, dim=0)
                 y = nd.array(y).T.reshape((-1,))
                 l = loss(outputs, y).mean()
             l.backward()
             grad_clipping(params, clipping_theta, context)
             utils.sgd(params, learning_rate, 1)
             l_sum += l.asscalar() * y.size
             n += y.size
         if (epoch + 1) % predict_period == 0:
             print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(l_sum / n), time.time() - start))
             for prefix in prefixes:
                 print('-', predict(prefix, predict_length, rnn, params, init_rnn_state, num_hiddens, vocabulary_size, context, index_to_char, char_to_index))




if __name__ == "__main__":
    chars, index_to_char, char_to_index, vocabulary_size = read_lyrics()
    num_inputs = vocabulary_size
    num_hiddens = 256
    num_output = vocabulary_size
    context = utils.try_gpu()
    num_epochs = 250
    num_steps = 35
    batch_size = 32
    learning_rate = 1e2
    clipping_theta = 1e-2
    predict_period = 1
    predict_length = 50
    prefixes = ['分开', '不分开']
    params = get_params()
    train(rnn,
          get_params,
          init_rnn_state,
          num_hiddens,
          vocabulary_size,
          context,
          chars,
          index_to_char,
          char_to_index,
          True,
          num_epochs,
          num_steps,
          learning_rate,
          clipping_theta,
          batch_size,
          predict_period,
          predict_length,
          prefixes)