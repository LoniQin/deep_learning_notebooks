from mxnet import nd
from mxnet.gluon import nn
from layers.DenseBlock import DenseBlock
def transition_block(num_channels):
    block = nn.Sequential()
    block.add(nn.BatchNorm(), nn.Activation('relu'), nn.Conv2D(num_channels, kernel_size=1), nn.AvgPool2D(pool_size=2, strides=2))
    return block

class DenseNet(nn.Sequential):
    def __init__(self, num_outputs, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3), nn.BatchNorm(), nn.Activation('relu'), nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_block = [4, 4, 4, 4]
        for i, num_convs in enumerate(num_convs_in_dense_block):
            self.add(DenseBlock(num_convs, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(num_convs_in_dense_block) - 1:
                num_channels //= 2
                self.add(transition_block(num_channels))
        self.add(nn.BatchNorm(), nn.Activation('relu'), nn.GlobalAvgPool2D(), nn.Dense(num_outputs))



if __name__ == "__main__":
    X = nd.random.uniform(shape=(2, 3,96,96))
    net = DenseNet(num_outputs=10)
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'ouput shape:', X.shape)