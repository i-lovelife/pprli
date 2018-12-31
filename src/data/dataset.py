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
    _default_type = 'ferg'
    def __init__(self, data=None, split=0.85, max_ele=1e9, transform=False, train_data=None, test_data=None):
        """
        data:
            {
                'x':[],
                'y':[]
            }
        """
        if data is not None:
            self.data = data
            length = min(max_ele, get_length(data))
            num_train = int(length * split)
            all_idx = np.random.permutation(length)
            self.train_data = {key: value[all_idx[:num_train]] for key, value in data.items()}
            self.test_data = {key: value[all_idx[num_train:length]] for key, value in data.items()}
        else:
            self.train_data = train_data
            self.test_data = test_data
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
    def get_train_len(self):
        return get_length(self.train_data)
    def get_test_len(self):
        return get_length(self.test_data)
    
    @classmethod
    def from_hp(cls, hp):
        type = hp.pop("type", cls._default_type)
        from_hdf5 = hp.pop("from_hdf5", True)
        if from_hdf5:
            return cls.by_name(type).from_hdf5(**hp)
        return cls.by_name(type)(**hp)


@Dataset.register('ferg')
class Ferg(Dataset):
    @classmethod
    def from_hdf5(cls,
                  path=DATA_ROOT / "processed/ferg.hdf5", 
                  select_people=[0, 1, 2, 3, 4, 5],
                  transform=True,
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
        return cls(data=data, transform=transform, **args)
    
    def process(self, data):
        x, y, p = data['x'], data['y'], data['p']
        x = x.astype('float32') / 255 - 0.5
        y = to_categorical(y, num_classes=7)
        p = to_categorical(p, num_classes=6)
        return {'x':x, 'y':y, 'p':p}
    
    def de_process(self, data):
        x, y, p = data['x'], data['y'], data['p']
        x = (x + 1) / 2 * 255
        x = np.round(x, 0).astype(int)
        y = np.argmax(y, axis=-1)
        p = np.argmax(p, axis=-1)
        return {'x':x, 'y':y, 'p':p}
    
    def transform(self):
        if self.transformed is True:
            return
        self.train_data = self.process(self.train_data)
        self.test_data = self.process(self.test_data)
        self.transformed = True
        
    def sample(self, num=64, selected_y=[], selected_p=[]):
        # sample data from train
        if selected_y == []:
            selected_y = [0, 1, 2, 3, 4, 5, 6]
        if selected_p == []:
            selected_p = [0, 1, 2, 3, 4, 5]
            
        x_all, y_all, p_all = self.train_data['x'], self.train_data['y'], self.train_data['p']
        idx = []
        for i in range(len(x_all)):
            y = y_all[i]
            p = p_all[i]
            if np.argmax(y, axis=-1) in selected_y and np.argmax(p, axis=-1) in selected_p:
                idx.append(i)
                if num > 0 and len(idx) >= num:
                    break
                    
        return {'x':x_all[idx], 'y':y_all[idx], 'p':p_all[idx]}

@Dataset.register('ferg_zero_one')
class FergZeroOne(Ferg):
    @classmethod
    def from_hdf5(cls,
                  path=DATA_ROOT / "processed/ferg.hdf5", 
                  select_people=[0, 1, 2, 3, 4, 5],
                  transform=True,
                  **args):
        with h5py.File(path, 'r') as f:
            x, y, p = f['x'][:], f['y'][:], f['z'][:]
        idx = np.array([idx for idx in range(len(x)) if p[idx] in select_people])
        remap_person = {x:i for i, x in enumerate(select_people)}
        x = x[idx]
        y = y[idx]
        p = p[idx]
        p = np.array([remap_person[person] for person in p])
        p = (p>0).astype(np.int32)
        data ={
            'x': x,
            'y': y,
            'p': p
        }
        return cls(data=data, transform=transform, **args)
    def process(self, data):
            x, y, p = data['x'], data['y'], data['p']
            x = x.astype('float32') / 255 - 0.5
            y = to_categorical(y, num_classes=7)
            p = to_categorical(p, num_classes=2)
            return {'x':x, 'y':y, 'p':p}
 
def test_ferg():
    ferg_full = Dataset.by_name('ferg').from_hdf5(select_people=[0, 1, 2, 3, 4, 5])
    train_data = ferg_full.train_data
    ferg_full.show_info()
          
    ferg_full = Dataset.by_name('ferg').from_hdf5(transform=False, select_people=[0, 1, 2, 3, 4, 5])
    train_data = ferg_full.train_data
    print(f'max_p = {np.max(train_data["p"])}, min_p={np.min(train_data["p"])}')

def test_ferg_zero_one():
    ferg_zero_one = Dataset.by_name('ferg_zero_one').from_hdf5()
    ferg_zero_one.show_info()

if __name__ == "__main__":
    test_ferg()
    test_ferg_zero_one()

