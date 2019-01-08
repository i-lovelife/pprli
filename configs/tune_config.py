import click
from src import EXPERIMENT_ROOT
from configs import Config

def tune_config(type, add_name=''):
    configs = Config.by_name(type).tune_config()
    EXPERIMENT_DIR = EXPERIMENT_ROOT / f'{type}-{add_name}-tune'
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    configs_file_path = EXPERIMENT_DIR / f'configs.list'
    configs_file = configs_file_path.open('w')
    for name, config in configs:
        experiment_dir = EXPERIMENT_ROOT / f'{name}'
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config.save(EXPERIMENT_ROOT / f'{name}' / 'config.json')
        configs_file.write(f'{name}\n')
    return len(configs)
    
@click.command()
@click.option('--name', type=str, default='')
@click.option('--type', type=str)
def main(name, type):
    config = tune_config(type, name)

if __name__=='__main__':
    main()
