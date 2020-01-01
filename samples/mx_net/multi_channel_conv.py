from utils import conv_2d
from mxnet import nd
def conv_2d_multi_in(X, K):
    return nd.add_n(*[conv_2d(x, k) for x, k in zip(X, K)])
# 2 * 3 * 3
X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
# 2 * 2 * 2
K = nd.array([[[0, 1], [2, 3]],
              [[1, 2], [3, 4]]])
items = []
for x, k in zip(X, K):
    items.append(conv_2d(x, k))
print(nd.add_n(X, X))
print(nd.add_n(*[X, X]))
print(nd.add_n(*items))
print(conv_2d_multi_in(X, K))

def conv_2d_multi_in_out(X, K):
    return nd.stack(*[conv_2d_multi_in(X, k) for k in K])

print(K)
print(nd.stack(K, K+1, K+2).shape)
print(conv_2d_multi_in_out(X, [K, K+1, K+2]))


def conv2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)
    return Y.reshape((c_o, h, w))
X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))
c_i, h, w = X.shape
c_o = K.shape[0]
print("X:", X, " shape:", X.shape)
X = X.reshape((c_i, h * w))
print("X: ", X, " shape:", X.shape)
K = K.reshape((c_o, c_i))
print("K: ", K, " shape:", K.shape)
Y = nd.dot(K, X)
print("Y: ", Y, " shape:", Y.shape)
Y = Y.reshape((c_o, h, w))
print("Y: ", Y, " shape:", Y.shape)
print("reshaped Y shape:", Y.shape)
X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))
Y = conv2d_multi_in_out_1x1(X, K)
print("Y: ", Y, " shape:", Y.shape)

X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))
Y1 = conv2d_multi_in_out_1x1(X, K)
Y2 = conv_2d_multi_in_out(X, K)
print("Y1 shape:", Y1.shape)
print("Y2 shape:", Y2.shape)
result = (Y1 - Y2).norm().asscalar()
print(result)