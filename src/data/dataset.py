import h5py
import click
import numpy as np
from keras.utils import to_categorical
from src.data.util import shuffle_data
from src.util.registerable import Registerable
from src import DATA_ROOT
class Dataset(Registerable):
    def __init__(self):
        pass
    def gen_batch(self, batch_size):
        pass

@Dataset.register('ferg_double')
class FergDouble(Dataset):
    def __init__(self, data):
        self.data = data
    @classmethod
    def from_hdf5(cls, path):
        with h5py.File(processed_data_path, 'r') as f:
            x, y, p = f['x'][:], f['y'][:], f['z'][:]
    def gen_batch(self, batch_size):
class DatasetLoader:
    def __init__(self, processed_data_path, num_y, num_p):
        with h5py.File(processed_data_path, 'r') as f:
            x_all, y_all, p_all = f['x'].value, f['y'].value, f['z'].value
        self.data_all = (x_all, y_all, p_all)
        self.num_y = num_y
        self.num_p = num_p
    
    def load_data(self, data_split=0.85, shuffle=True, max_train=int(1e9), max_test=int(1e9)):
        data_all = shuffle_data(self.data_all) if shuffle is True else self.data_all
        (x_all, y_all, p_all) = data_all
        num_train = int(x_all.shape[0] * data_split)
        num_train = min(num_train, max_train)
        num_test = x_all.shape[0] - num_train
        num_test = min(num_test, max_test)
        x_train, x_test = x_all[:num_train], x_all[num_train:num_train+num_test]
        y_train, y_test = y_all[:num_train], y_all[num_train:num_train+num_test]
        p_train, p_test = p_all[:num_train], p_all[num_train:num_train+num_test]
        return (x_train, y_train, p_train), (x_test, y_test, p_test)
    def get_num_classes(self):
        return self.num_y, self.num_p

def load_ferg():
    return DatasetLoader(DATA_ROOT / "processed/ferg.hdf5", 7, 6)
def test():
    processed_data_path = DATA_ROOT / "processed/ferg.hdf5"
    num_y = 7
    num_p = 6
    ferg_loader = DatasetLoader(processed_data_path, num_y, num_p)
    train_data, test_data = ferg_loader.load_data()
    (x_train, y_train, p_train), (x_test, y_test, p_test) = (train_data, test_data)
    assert(x_train.shape == (47401, 64, 64, 3))
    assert(y_train.shape == (47401, 7))
    assert(p_train.shape == (47401, 6))
    assert(x_test.shape == (8365, 64, 64, 3))
    assert(y_test.shape == (8365, 7))
    assert(p_test.shape == (8365, 6))

if __name__ == "__main__":
    test()

