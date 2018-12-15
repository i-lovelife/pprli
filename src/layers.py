from src.util.registerable import Registerable
from keras import Model

class ModelBuilder(Registerable):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layers(inputs)
        return inputs

@LayerBuilder.register('cnn')
class CnnBuilder(ModelBuilder):
    def __init__(num_layers=3,
                 max_num_channels=64*3,
                 f_size=4):
        """
        Input: (img_dim, img_dim, 3)
        """
        super(CnnBuilder, self).__init__()

        for i in range(num_layers + 1):
            num_channels = max_num_channels // 2**(num_layers - i)
            conv2d = Conv2D(num_channels,
                       (5, 5),
                       strides=(2, 2),
                       use_bias=False,
                       padding='same')
            self.layers.push(conv2d)
            if i > 0:
                self.layers.push(BatchNormalization())
            self.add_layer(LeakyReLU(0.2))
        self.add_layer(Flatten())

