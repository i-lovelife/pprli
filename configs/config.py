import ujson as json
from src.util.registerable import Registerable
class Config(Registerable):
    def __init__(self, config):
        self.config = config

    @classmethod
    def make_config(self, NAME):
        raise NotImplementedError

    def save(self, path):
        with path.open(mode='w') as f:
            json.dump(self.config, f)
