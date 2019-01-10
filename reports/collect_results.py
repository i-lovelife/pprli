import ujson as json
import click

from src import EXPERIMENT_ROOT, REPORT_ROOT


"""
epochs:0 UtilityTaskEvaluater_0: 0.18000000059604646 PrivateTaskEvaluater_1: 0.49333333532015483
epochs:1 UtilityTaskEvaluater_0: 0.19333333452542623 PrivateTaskEvaluater_1: 0.4200000011920929
"""

privacy_range = [0.2, 0.3, 0.4, 0.5, 0.6]
utility_range = [0.5, 0.6, 0.7, 0.8, 0.9]
def analyze(config_names):
    for name in config_names:
        name = name.strip()
        experiment_dir = EXPERIMENT_ROOT / f'{name}'
        result_path = experiment_dir / 'result.txt'
        config_path = experiment_dir / 'config.json'
        
        
        config_file = config_path.open('r')
        config = json.load(config_file)
        algo = config["privater"]["type"]
        record_path = REPORT_ROOT / f'{algo}.json'
        if not record_path.exists():
            file = record_path.open('w')
            json.dump({}, file)
            file.close()
            record = {}
        else:
            record = json.load(record_path.open('r'))
            
        all_lines = result_path.open().readlines()
        for line in all_lines:
            line = line.strip()
            u_value = float(line.split()[2])
            p_value = float(line.split()[4])
            for privacy_threshold in privacy_range:
                best_before = record.get(f'privacy{privacy_threshold:.2f}', -1)
                if p_value < privacy_threshold and u_value > best_before:
                    record[f'privacy{privacy_threshold:.2f}'] = u_value
                    
            for utility_threshold in utility_range:
                best_before = record.get(f'utility{utility_threshold:.2f}', 2)
                if u_value >  utility_threshold and p_value < best_before:
                    record[f'utility{utility_threshold:.2f}'] = p_value
        
        record_file = record_path.open('w')
        json.dump(record, record_file)
        record_file.close()
        print(algo)
        print(record)
                

@click.command()
@click.option('--config_list', type=str, default=None)
@click.option('--config', type=str, default=None)
def main(config_list, config):
    if config_list is not None:
        config_list_path = EXPERIMENT_ROOT / config_list
        config_names = config_list_path.open('w').readlines()
    else:
        config_names = [config]
    analyze(config_names)

if __name__ == '__main__':
    main()