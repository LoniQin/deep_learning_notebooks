from mxnet import nd
import random
import zipfile
def read_lyrics():
    zfile = zipfile.ZipFile('../../data/jaychou_lyrics.zip')
    data = zfile.read(zfile.namelist()[0])
    string = str(data, encoding='utf8')
    chars = string.replace('\n', ' ').replace('\r', ' ')
    # Create index
    index_to_char = []
    for char in chars:
        if index_to_char.__contains__(char) == False:
            index_to_char.append(char)
    char_to_index = dict([(value, key) for key, value in enumerate(index_to_char)])
    indices = [char_to_index[char] for char in chars]
    return indices, index_to_char, char_to_index, len(index_to_char)


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
if __name__ == "__main__":
    print(read_lyrics())