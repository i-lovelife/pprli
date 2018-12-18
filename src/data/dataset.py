import h5py
import click
import numpy as np
from keras.utils import to_categorical
from src.data.util import shuffle_data
from src.util.registerable import Registerable
from src import DATA_ROOT

def get_length(data):
    return next(iter(data.items()))[1].shape[0]
class Dataset(Registerable):
    def __init__(self, data, split=0.85, max_ele=1e9, transform=False, **args):
        """
        data:
            {
                'x':[],
                'y':[]
            }
        """
        self.data = data
        length = min(max_ele, get_length(data))
        num_train = int(length * split)
        all_idx = np.random.permutation(length)
        self.train_data = {key: value[all_idx[:num_train]] for key, value in data.items()}
        self.test_data = {key: value[all_idx[num_train:length]] for key, value in data.items()}
        
        self.transformed = False
        if transform:
            self.transform()

    def transform(self):
        self.transformed = True
        
    def get_data(self, data, num=-1):
        length = get_length(data)
        if num<0:
            num = length
        idx = np.random.choice(length, num, replace=False)
        ret = {key: value[idx] for key, value in data.items()}
        return ret

    def get_shape(self, data, key=None):
        if key is None:
            for key, value in data.items():
                print(f'{key}:{value.shape}')
        else:
            print(f'{key}:{data[key].shape}')

    def show_info(self):
        print('train data')
        self.get_shape(self.train_data)
        print('test data')
        self.get_shape(self.test_data)

    def get_train(self):
        return self.train_data
    def get_test(self):
        return self.test_data
    def get_train_batch(self, batch_size):
        return self.get_data(self.train_data, num=batch_size)
    def get_test_batch(self, batch_size):
        return self.get_data(self.test_data, num=batch_size)


#@Dataset.register('ferg')
class Ferg(Dataset):
    @classmethod
    def from_hdf5(cls,
                  path=DATA_ROOT / "processed/ferg.hdf5", 
                  select_people=[0, 1],
                  **args):
        with h5py.File(path, 'r') as f:
            x, y, p = f['x'][:], f['y'][:], f['z'][:]
        idx = np.array([idx for idx in range(len(x)) if p[idx] in select_people])
        remap_person = {x:i for i, x in enumerate(select_people)}
        x = x[idx]
        y = y[idx]
        p = p[idx]
        p = np.array([remap_person[person] for person in p])
        data ={
            'x': x,
            'y': y,
            'p': p
        }
        return cls(data, **args)
    
    def transform(self):
        if self.transformed is True:
            return
        def process(data):
            x, y, p = data['x'], data['y'], data['p']
            x = x.astype('float32') / 255 - 0.5
            y = to_categorical(y, num_classes=7)
            p = to_categorical(p, num_classes=6)
            return {'x':x, 'y':y, 'p':p}
        self.train_data = process(self.train_data)
        self.test_data = process(self.test_data)
        self.transformed = True

#@Dataset.register('ferg_zero_one')
class FergZeroOne(Dataset):
    @classmethod
    def from_hdf5(cls,
                  path=DATA_ROOT / "processed/ferg.hdf5", 
                  select_people=[0, 1],
                  **args):
        if len(set(select_people))!=2:
            raise ValueError(f'len of {select_people} is not 2')
        with h5py.File(path, 'r') as f:
            x, y, p = f['x'][:], f['y'][:], f['z'][:]
        x0_idx = np.array([idx for idx in range(len(x)) if p[idx] == select_people[0]])
        x1_idx = np.array([idx for idx in range(len(x)) if p[idx] == select_people[1]])
        x0 = x[x0_idx]
        x1 = x[x1_idx]
        y0 = y[x0_idx]
        y1 = y[x1_idx]
        data ={
            'x0': x0,
            'x1': x1,
            'y0': y0,
            'y1': y1
        }
        return cls(data, **args)
 
def test_ferg():
    ferg_full = Dataset.by_name('ferg').from_hdf5(select_people=[0, 1, 2, 3, 4, 5])
    train_data = ferg_full.train_data
    ferg_full.show_info()
    print(f'max_p = {np.max(train_data["p"])}, min_p={np.min(train_data["p"])}')

def test_ferg_zero_one():
    ferg_zero_one = Dataset.by_name('ferg_zero_one').from_hdf5()
    ferg_zero_one.show_info()

if __name__ == "__main__":
    #test_ferg()
    test_ferg_zero_one()

