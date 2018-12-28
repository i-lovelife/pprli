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
                 privater):
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