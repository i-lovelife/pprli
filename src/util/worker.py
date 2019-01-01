from src.util.registerable import Registerable
from src.optimizer import Optimizer
class Worker(Registerable):
    def __init__(self, optimizer={}):
        self.optimizer = Optimizer.from_hp(optimizer)