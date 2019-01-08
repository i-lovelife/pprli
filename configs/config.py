import ujson as json
from src.util.registerable import Registerable
class Config(Registerable):
    def __init__(self, config):
        self.config = config

    @classmethod
    def make_config(cls):
        raise NotImplementedError

    @classmethod
    def tune_config(cls):
        pass

    def save(self, path):
        with path.open(mode='w') as f:
            json.dump(self.config, f)
