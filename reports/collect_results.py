import ujson as json
import click

from src import EXPERIMENT_ROOT


"""
epochs:0 UtilityTaskEvaluater_0: 0.18000000059604646 PrivateTaskEvaluater_1: 0.49333333532015483
epochs:1 UtilityTaskEvaluater_0: 0.19333333452542623 PrivateTaskEvaluater_1: 0.4200000011920929
"""
def analyze(config_names):
    results = []
    for name in config_names:
        name = name.strip()
        experiment_dir = EXPERIMENT_ROOT / f'{name}'
        result_path = experiment_dir / 'result.txt'
        config_path = experiment_dir / 'config.json'
        f = result_path.open('r')
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            u_value = float(line.split()[2])
            p_value = float(line.split()[4])
            results.append((u_value, p_value))
    return results

@click.command
@click.option('--config_list', type=str, default=None)
@click.option('--config_single', type=str, default=None)
def main(config_list, config_single):
    if config_list is not None:
        config_list_path = EXPERIMENT_ROOT / config_list
        config_names = config_list_path.open('w').readlines()
    else:
        config_names = [config_single]
    analyze(config_names)

if __name__ == '__main__':
    main()
