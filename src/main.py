import click
import ujson as json

def train(x_train, 
          y_train,
          x_test,
          y_test):
    pass

def test(x_train,
         y_train,
         x_test,
         y_test):
    pass

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    debug = config['debug']
    dataset = Dataset.by_name(config['dataset'])(debug=debug)
    x_train, x_test, y_train, y_test = dataset.load_data()

if __name__ == "__main__":
    main()
