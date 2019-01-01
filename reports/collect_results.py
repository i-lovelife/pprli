from src import EXPERIMENT_ROOT
def collect_result(prefix=''):
    all_path = sorted(EXPERIMENT_ROOT.glob(f'{prefix}*/stdout.log'))
    print(len(all_path))
    for path in all_path:
        config_path = path.parent / 'config.json'

if __name__ == '__main__':
    collect_result()
