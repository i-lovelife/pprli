import click
from src import EXPERIMENT_ROOT
from configs import Config

def make_config(type, name=None):
    if name is None:
        name = type
    config_cls = Config.by_name(type)
    config_ins = config_cls.make_config()
    config_ins.save(EXPERIMENT_ROOT / f'{name}' / 'config.json')
    return config_ins.config
    
@click.command()
@click.option('--type', type=str)
@click.option('--name', type=str, default=None)
def main(type, name):
    config = make_config(type, name)

if __name__=='__main__':
    main()
