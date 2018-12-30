import numpy as np
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
from src.data.dataset import Dataset
from src.trainer import KerasTrainer
from src.util.worker import Worker
from src.callbacks import EarlyStopping


class Evaluater(Worker):
    _default_type='private'
    def build_model(self, data):
        raise NotImplementedError
    
    def get_input(self, data):
        """return (x, y)
        """
        raise NotImplementedError
        
    def summary(self):
        model = self.build_model()
        model.summary()
        
    def evaluate(self, 
                 dataset, 
                 privater):
        raise NotImplementedError
        
    def save_weights(self, path):
        self.train_model.save_weights(path)
        
    def load_weights(self, path):
        self.train_model.save_weights(path)

@Evaluater.register('reconstruction')
class ImgReconstructionEvaluater(Evaluater):
    def __init__(self,
                 base_dir=None,
                 img_dim=64,
                 selected_y=[0, 1, 2, 3, 4, 5, 6],
                 selected_p=[0, 1, 2, 3, 4, 5]):
        if base_dir is not None:
            base_dir = base_dir/type(self).__name__
            base_dir.mkdir(exist_ok=True)
          
        self.selected_y = selected_y
        self.selected_p = selected_p
        self.base_dir = base_dir
        self.img_dim = img_dim
    
    def build_model(self, data):
        pass
    def get_input(self, data):
        pass
    def summary(self):
        pass
    def evaluate(self,
                 dataset,
                 privater,
                 epoch):
        if self.base_dir is None:
            return
        img_dim = self.img_dim
        selected_p = self.selected_p
        selected_y = self.selected_y
        img_path=f'{str(self.base_dir/str(epoch))}.png'
        figure = np.zeros((img_dim * len(selected_p), img_dim * len(selected_y), 3))
        for i, p in enumerate(selected_p):
            for j, y in enumerate(selected_y):
                data = dataset.sample(selected_y=[y], selected_p=[p], num=1)
                rec_data = privater.reconstruct(data)
                rec_data = dataset.de_process(rec_data)
                figure[i * img_dim:(i + 1) * img_dim,
                       j * img_dim:(j + 1) * img_dim] = rec_data['x']
        imageio.imwrite(img_path, figure)
        
@Evaluater.register('latent_visual')
class LatentVisualizationEvaluater(ImgReconstructionEvaluater):
    def evaluate(self,
                 dataset,
                 privater,
                 epoch):
        if self.base_dir is None:
            return
        img_dim = self.img_dim
        selected_p = self.selected_p
        selected_y = self.selected_y
        img_path=f'{str(self.base_dir/str(epoch))}.png'
        figure = np.zeros((img_dim * len(selected_p), img_dim * len(selected_y), 3))
        for i, p in enumerate(selected_p):
            for j, y in enumerate(selected_y):
                data = dataset.sample(selected_y=[y], selected_p=[p], num=1)
                rec_data = privater.reconstruct(data)
                rec_data = dataset.de_process(rec_data)
                figure[i * img_dim:(i + 1) * img_dim,
                       j * img_dim:(j + 1) * img_dim] = rec_data['x']
        imageio.imwrite(img_path, figure)

class DownTaskEvaluater(Evaluater):
    def __init__(self,
                 num_classes=7,
                 z_dim=128,
                 epochs=20,
                 batch_size=64,
                 verbose=False):
        self.num_classes= num_classes
        self.z_dim = z_dim
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
    def build_model(self):
        num_classes, z_dim = self.num_classes, self.z_dim
        x_in = Input(shape=(z_dim,))
        x = x_in
        x = Dense(z_dim, activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(x_in, x)
        model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['acc'])
        return model
    def evaluate(self,
                 dataset,
                 privater,
                 epoch):
        print(f'evaluating {type(self).__name__} on epoch {epoch}')
        self.train_model = self.build_model()
        train_data = dataset.get_train()
        test_data = dataset.get_test()
        train_data = privater.predict(train_data)
        test_data = privater.predict(test_data)
        dataset = Dataset(train_data=train_data, test_data=test_data)
        early_stopping = EarlyStopping()
        callbacks = [early_stopping]
        trainer = KerasTrainer(batch_size=self.batch_size, epochs=self.epochs)
        trainer.train(dataset, self, callbacks=callbacks)
        return min(early_stopping.best_val_acc, early_stopping.best_acc)
    
@Evaluater.register('private')
class PrivateTaskEvaluater(DownTaskEvaluater):
    def __init__(self, z_dim=128, **args):
        super().__init__(z_dim=z_dim, num_classes=6, **args)
        
    def get_input(self, data):
        return (data['x'], data['p'])
    
@Evaluater.register('utility')
class UtilityTaskEvaluater(DownTaskEvaluater):
    def __init__(self, z_dim=128, **args):
        super().__init__(z_dim=z_dim, num_classes=7, **args)
        
    def get_input(self, data):
        return (data['x'], data['y'])