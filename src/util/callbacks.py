from keras.callbacks import Callback
from pathlib import Path
from src import EXPERIMENT_ROOT

class Evaluate(Callback):
    def __init__(self, task, sample_dir=None, model_path=None):
        import os
        self._task = task
        self.lowest = 1e10
        self.losses = []
        if sample_dir is None:
            sample_dir = EXPERIMENT_ROOT / type(task).__name__ / 'sample'
        else:
            sample_dir = Path(sample_dir)
        if model_path is None:
            model_path = EXPERIMENT_ROOT / type(task).__name__ / 'best.h5'
        else:
            model_path = Path(model_path)
        sample_dir.mkdir(parents=True, exist_ok=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_dir = sample_dir
        self.model_path = model_path
    def on_epoch_end(self, epoch, logs=None):
        self._task.sample_all(self.sample_dir / f'{epoch}.png')
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            self._task.save_weights(self.model_path)