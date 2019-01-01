import click
import ujson as json
from src import EXPERIMENT_ROOT

z_dim=64
evaluation_verbose=False
class Config:
    def __init__(self, config):
        self.config  = config
    @classmethod
    def make_config(self, NAME)
        config={
            "privater":{
                "type":"cvae",
                "z_dim":z_dim,
                "rec_x_weight":64*64/10,
                "encrypt_with_noise": True,
                "optimizer":{
                    "type": "adam",
                    "lr":0.0001,
                }
            },
            "dataset":{
                "type":"ferg"
            },
            "trainer":{
                "type":"keras",
                "epochs":100
            },
            "evaluaters":[
                {"type":"utility",
                 "z_dim":z_dim,
                 "verbose": evaluation_verbose
                },
                {"type":"private",
                 "z_dim":z_dim,
                 "verbose": evaluation_verbose
                },
                {"type":"reconstruction",
                 "base_dir": NAME
                }
            ]
        }
        return Config(config)
    def save(self, path):
        with path.open(mode='w') as f:
            json.dump(self.config, f)

@click.Command
@click.option('--name', type=str)
def main(name):
    config = Config.make_config(name)
    config.save(EXPERIMENT_ROOT / f'{name}' / 'config.json')

if __name__=='__main__':
    main()
