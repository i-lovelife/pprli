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
from shutil import rmtree
import ujson as json
import click

from src.data.dataset import Dataset
from src.trainer import Trainer
from src.evaluater import Evaluater
from src.privater import Privater
from src import CONFIG_ROOT, EXPERIMENT_ROOT
from src.util.logging import TeeLogger

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
        
        sys.stdout = TeeLogger(log_path, sys.stdout)
        sys.stderr = TeeLogger(log_path, sys.stderr)
    else:
        config = {}
    
    privater_config = config.pop('privater', {'type': 'vae'})
    privater = Privater.from_hp(privater_config)
    privater.summary()
    if show:
        return
    
    dataset_config = config.pop('dataset', {'type': 'ferg'})
    if debug:
        dataset_config['max_ele'] = 1000
    dataset = Dataset.from_hp(dataset_config)
    dataset.show_info()
    
    trainer_config = config.pop('trainer', {'type': 'keras'})
    trainer = Trainer.from_hp(trainer_config)
    evaluaters_config = config.pop('evaluaters', [{'type': 'utility'}, {'type': 'private'}])
    evaluaters = [Evaluater.from_hp(evaluater_config) for evaluater_config in evaluaters_config]
    historys = trainer.train(dataset, worker=privater, evaluaters=evaluaters)
    for history in historys:
        print(history)
if __name__ == '__main__':
    main()