import click
import ujson as json
from _jsonnet import evaluate_file, evaluate_snippet
from src.data.dataset import DatasetLoader
from src.models import Model
import os
@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('config_path', type=click.Path(exists=True))
def main(data_path, config_path):
    config = json.loads(evaluate_file(config_path))
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    data_loader = DatasetLoader(data_path, config['num_y'], config['num_z'])
    data = data_loader.load_data()
    model = Model.by_name(config['type'])(config)
    acc1, acc2 = model.evaluate(data)
    print(f'acc1 = {acc1}, acc2= {acc2}')

if __name__ == '__main__':
    main()