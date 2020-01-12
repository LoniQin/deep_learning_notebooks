from mxnet import nd
import random
def read_lyrics():
    f = open('../../data/jaychou_lyrics.txt')
    chars = f.read().replace('\n', ' ').replace('\r', ' ')
    # 建立索引
    index_to_char = list(set(chars))
    char_to_index = dict([(value, key) for key, value in enumerate(index_to_char)])
    return chars, index_to_char, char_to_index, len(index_to_char)


def data_iter(chars, batch_size, num_steps, is_random = False, ctx = None):
    num_examples = (len(chars) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    if is_random:
        random.shuffle(example_indices)
    for i in range(epoch_size):
        batch_indices = example_indices[i * batch_size: (i + 1) * batch_size]
        X = [chars[j * num_steps: (j + 1) * num_steps] for j in batch_indices]
        Y = [chars[j * num_steps + 1: (j + 1) * num_steps + 1] for j in batch_indices]
        yield X, Y

def grad_clipping(params, theta, context):
    norm = nd.array([0], context)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm