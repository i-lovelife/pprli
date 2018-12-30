import numpy as np
import glob
import imageio
from keras.models import Model
from keras.models import load_model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


import os
import sys
from shutil import rmtree, copyfile
import ujson as json
import click

from src.data.dataset import Dataset
from src.trainer import Trainer
from src.evaluater import Evaluater
from src.privater import Privater
from src import CONFIG_ROOT, EXPERIMENT_ROOT
from src.util.tee_logging import TeeLogger
from src.callbacks import EvaluaterCallback

@click.command()
@click.option('--name', default=None)
@click.option('--show/--no-show', default=False)
@click.option('--debug/--no-debug', default=False)
@click.option('--gpu', type=click.Choice(['0', '1', '2', '3']), default='0')
def main(name, show, debug, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if name is not None:
        config_path = CONFIG_ROOT / (name+'.config')
        with config_path.open() as f:
            config = json.load(f)
        experiment_path = EXPERIMENT_ROOT / name
        if experiment_path.exists():
            rmtree(experiment_path)
        experiment_path.mkdir()
        
        log_path = experiment_path / 'stdout.log'
        copyfile(config_path, experiment_path/ (name+'.config'))
        
        sys.stdout = TeeLogger(log_path, sys.stdout)
        sys.stderr = TeeLogger(log_path, sys.stderr)
    else:
        name = 'tmp'
        config = {}
    
    privater_config = config.pop('privater', 
                                 {'type': 'gpf',
                                  'z_dim': 64,
                                  'fake_rec_z_weight':100
                                 })
    privater = Privater.from_hp(privater_config)
    privater.summary()
    if show:
        return
    
    dataset_config = config.pop('dataset', {'type': 'ferg'})
    if debug:
        dataset_config['max_ele'] = 1000
    dataset = Dataset.from_hp(dataset_config)
    dataset.show_info()
    
    trainer_config = config.pop('trainer', {'type': 'adv', 'epochs':50})
    trainer = Trainer.from_hp(trainer_config)
    evaluaters_config = config.pop('evaluaters', [{'type': 'utility', 'z_dim':64}, 
                                                  {'type': 'private', 'z_dim':64},
                                                  {'type': 'reconstruction', 'base_dir':EXPERIMENT_ROOT / name}
                                                 ])
    callbacks = [EvaluaterCallback(Evaluater.from_hp(evaluater_config), dataset, privater) for evaluater_config in evaluaters_config]
    historys = trainer.train(dataset, worker=privater, callbacks=callbacks)
    for i in range(trainer.epochs):
        output = ' '.join([f'{key}: {value}' for key, value in history.items()])
        print(f'epochs:{i} {output}')
if __name__ == '__main__':
    main()