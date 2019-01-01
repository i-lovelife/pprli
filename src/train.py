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
import importlib


@click.command()
@click.option('--name', default=None)
@click.option('--show/--no-show', default=False)
@click.option('--debug/--no-debug', default=False)
@click.option('--hpc/--no-hpc', default=False)
@click.option('--gpu', type=click.Choice(['0', '1', '2', '3']), default='0')
def main(name, show, debug, gpu, hpc):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if name is not None:
        if hpc:
            config_mod = importlib.import_module(f'experiments.{name}.{name}')
        else:
            config_mod = importlib.import_module('configs.'+name)
        config = config_mod.config
        experiment_path = EXPERIMENT_ROOT / name
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        log_path = experiment_path / 'stdout.log'
        if log_path.exists():
            log_path.unlink()
        copyfile(CONFIG_ROOT/f'{name}.py', experiment_path/f'{name}.py')
        
        sys.stdout = TeeLogger(log_path, sys.stdout)
        sys.stderr = TeeLogger(log_path, sys.stderr)
    else:
        name = 'tmp'
        config = {}
    
    privater_config = config.pop('privater', 
                                 {'type': 'gpf_zero_one',
                                  'p_dim': 2,
                                  'z_dim': 64,
                                  'fake_rec_z_weight':100,
                                  'real_rec_x_weight':0
                                 })
    privater = Privater.from_hp(privater_config)
    privater.summary()
    if show:
        return
    
    dataset_config = config.pop('dataset', {'type': 'ferg_zero_one'})
    if debug:
        dataset_config['max_ele'] = 1000
    dataset = Dataset.from_hp(dataset_config)
    dataset.show_info()
    
    trainer_config = config.pop('trainer', {'type': 'adv', 'epochs':50})
    if debug:
        trainer_config['epochs'] = 2
    trainer = Trainer.from_hp(trainer_config)
    evaluaters_config = config.pop('evaluaters', [{'type': 'utility', 'z_dim':64}, 
                                                  {'type': 'private', 'z_dim':64, 'num_classes':2},
                                                  {'type': 'reconstruction', 'base_dir':name, 'selected_p':[0, 1]}
                                                 ])
    callbacks = [EvaluaterCallback(Evaluater.from_hp(evaluater_config), dataset, privater) for evaluater_config in evaluaters_config]
    historys = trainer.train(dataset, worker=privater, callbacks=callbacks)
    for i in range(trainer.epochs):
        output = ' '.join([f'{key}: {value[i]}' for key, value in historys.items()])
        print(f'epochs:{i} {output}')
if __name__ == '__main__':
    main()
