import click
import ujson as json
from _jsonnet import evaluate_file, evaluate_snippet
from src.models import Model
from src.data.dataset import DatasetLoader
@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('config_path', type=click.Path(exists=True))
def main(data_path, config_path):
    config = json.loads(evaluate_file(config_path))
	data_loader = DatasetLoader(data_path, config['num_y'], config['num_z'])
	data = data_loader.load_data()
	model = Model.by_name(config['type'])(config)
    model.train(data)

if __name__ == '__main__':
    main()