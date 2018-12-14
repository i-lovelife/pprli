from src.util.registerable import Registerable
from keras import Model

class ModelBuilder(Registerable):
    def __call__(self, inputs):
        raise NotImplementedError

@LayerBuilder.register('cnn')
class CnnBuilder(ModelBuilder):
    def __init__(num_layers=3,
                 max_num_channels=64*3,
                 f_size=4):
        x_in = Input(shape=(img_dim, img_dim, 3))
        x = x_in
        for i in range(num_layers + 1):
            num_channels = max_num_channels // 2**(num_layers - i)
            x = Conv2D(num_channels,
                       (5, 5),
                       strides=(2, 2),
                       use_bias=False,
                       padding='same')(x)
            if i > 0:
                x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        return Model(x_in, x))

    def __call__(self, inputs)
