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
from src import EXPERIMENT_ROOT


class Evaluater(Worker):
    _default_type='private'
    def __init__(self, **args):
        super().__init__(**args)
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
                 selected_p=[0, 1, 2, 3, 4, 5],
                 **args):
        super().__init__(**args)
        if base_dir is not None:
            base_dir = EXPERIMENT_ROOT / base_dir / type(self).__name__
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
    def __init__(self, num_each_classes=1000, **args):
        super().__init__(**args)
        self.num_each_classes = num_each_classes
        
    def evaluate(self,
                 dataset,
                 privater,
                 epoch):
        pass
    

class DownTaskEvaluater(Evaluater):
    def __init__(self,
                 num_classes=7,
                 epochs=20,
                 batch_size=64,
                 verbose=False,
                 **args):
        super().__init__(**args)
        self.num_classes= num_classes
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
    def build_model(self, data):
        num_classes, z_dim = self.num_classes, data['x'].shape[-1]
        x_in = Input(shape=(z_dim,))
        x = x_in
        x = Dense(z_dim, activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(x_in, x)
        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])
        return model
    def evaluate(self,
                 dataset,
                 privater,
                 epoch=-1):
        print(f'evaluating {type(self).__name__} on epoch {epoch}')
        train_data = dataset.get_train()
        test_data = dataset.get_test()
        train_data = privater.predict(train_data)
        test_data = privater.predict(test_data)
        dataset = Dataset(train_data=train_data, test_data=test_data)
        self.train_model = self.build_model(train_data)
        early_stopping = EarlyStopping()
        callbacks = [early_stopping]
        trainer = KerasTrainer(batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
        trainer.train(dataset, self, callbacks=callbacks)
        return min(early_stopping.best_val_acc, early_stopping.best_acc)
    
@Evaluater.register('private')
class PrivateTaskEvaluater(DownTaskEvaluater):
    def __init__(self, num_classes=6, **args):
        super().__init__(num_classes=num_classes, **args)
        
    def get_input(self, data):
        return (data['x'], data['p'])
    
@Evaluater.register('utility')
class UtilityTaskEvaluater(DownTaskEvaluater):
    def __init__(self, num_classes=7, **args):
        super().__init__(num_classes=num_classes, **args)
        
    def get_input(self, data):
        return (data['x'], data['y'])
    
def build_decoder(img_dim=64,
                  z_dim=128,
                  num_conv=3,
                  max_channels=512):
    f_size = img_dim // (2**(num_conv + 1))
    z_in = Input(shape=(z_dim,))
    z = z_in
    z = Dense(f_size**2*max_channels, activation='relu')(z)
    z = Reshape((f_size, f_size, max_channels))(z)
    for i in range(num_conv):
        channels = max_channels // 2**(i + 1)
        z = Conv2DTranspose(channels,
                            (5, 5),
                            strides=(2, 2),
                            padding='same')(z)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)
    z = Conv2DTranspose(3,
                        (5, 5),
                        strides=(2, 2),
                        padding='same',
                        activation='tanh')(z)
    return Model(z_in, z)

@Evaluater.register('ssim')
class Ssim(DownTaskEvaluater):
    def __init__(self, **args):
        super().__init__(**args)
    def build_model(self, data):
        z_dim = data['z'].shape[-1]
        decoder = build_decoder(z_dim=z_dim)
        decoder.compile(optimizer=self.optimizer, loss='mean_squared_error')
        return decoder
    def evaluate(self,
                 dataset,
                 privater,
                 epoch=-1):
        print(f'evaluating {type(self).__name__} on epoch {epoch}')
        train_data = dataset.get_train()
        test_data = dataset.get_test()
        def process(data):
            pred_data = privater.predict(data)
            return {'z':pred_data['x'],'x':data['x']}
        train_data = process(train_data)
        test_data = process(test_data)
        model = self.build_model(train_data)
        history=model.fit(*self.get_input(train_data),
                  epochs=self.epochs, 
                  batch_size=self.batch_size,
                  validation_data=self.get_input(test_data),
                  verbose= self.verbose)
        return np.min(history.history['val_loss'])
    
    def get_input(self, data):
        return (data['z'], data['x'])
    
@Evaluater.register('ndm')
class Ndm(DownTaskEvaluater):
    def __init__(self, **args):
        super().__init__(**args)
    def build_model(self, data):
        def binary_loss(args):
            score_real, score_fake = args
            return - K.mean(K.log(score_real + 1e-6) + K.log(1 - score_fake + 1e-6))
        def shuffling(x):
            idxs = K.arange(0, K.shape(x)[0])
            idxs = K.tf.random_shuffle(idxs)
            return K.gather(x, idxs)
        z_dim = data['z'].shape[-1]
        z_concat_in = Input(shape=(2*z_dim,))
        z = z_concat_in
        z = Dense(z_dim, activation='relu')(z)
        z = Dense(z_dim, activation='relu')(z)
        z = Dense(z_dim, activation='relu')(z)
        z = Dense(1, activation='sigmoid')(z)
        dis = Model(z_concat_in, z)
        z_in = Input(shape=(z_dim,))
        z_shuffle = Lambda(shuffling)(z_in)
        true_score = dis(Concatenate()([z_in, z_in]))
        fake_score = dis(Concatenate()([z_in, z_shuffle]))
        loss = Lambda(binary_loss)([true_score, fake_score])
        model = Model(z_in, loss)
        #model.summary()
        def identity_loss(y_true, y_pred):
            return y_pred
        model.compile(optimizer=self.optimizer, loss=identity_loss)
        return model
    def evaluate(self,
                 dataset,
                 privater,
                 epoch=-1):
        print(f'evaluating {type(self).__name__} on epoch {epoch}')
        train_data = dataset.get_train()
        test_data = dataset.get_test()
        def process(data):
            pred_data = privater.predict(data)
            return {'z':pred_data['x']}
        train_data = process(train_data)
        test_data = process(test_data)
        model = self.build_model(train_data)
        history=model.fit(*self.get_input(train_data),
                  epochs=self.epochs, 
                  batch_size=self.batch_size,
                  validation_data=self.get_input(test_data),
                  verbose= self.verbose)
        return np.min(history.history['val_loss'])
    
    def get_input(self, data):
        return (data['z'], data['z'])