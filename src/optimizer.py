from keras import optimizers
from src.util.registerable import Registerable

class Optimizer(Registerable):
    _default_type='adam'
    def __init__(self):
        pass

Registerable._registry[Optimizer] = {
        "adam": optimizers.Adam,
        "sgd": optimizers.SGD,
        "rmsprop": optimizers.RMSprop
}