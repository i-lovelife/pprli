import matplotlib.pyplot as plt
from keras.layers import Input, Lambda, BatchNormalization, Conv2D, Reshape, Dense,\
                         Dropout, Activation, Flatten, LeakyReLU, Add, MaxPooling2D,\
                         GlobalMaxPooling2D, Subtract, Concatenate, Average, Conv2DTranspose,\
                         GlobalAveragePooling2D
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.optimizers import RMSprop
from keras.metrics import binary_accuracy
from numpy.random import seed
from tensorflow import set_random_seed
from pathlib import Path
import sys
import logging
import time
import numpy as np
from src.data.dataset import load_ferg
from src.evaluation.resnet import resnet_v1
from src import PROJECT_ROOT, EXPERIMENT_ROOT
from src.util.logging import TeeLogger
import imageio
import os
from src.util.callbacks import Evaluate
from src.models.task import FergTask
import click
"""
pprl model
"""
class Pprl:
    def __init__(self,
                 img_dim=64,
                 z_dim,
                 kl_weight=-1.,
                 rec_weight=-1.,
                 encoder_share=False,
                 cls_y_weight=-1.,
                 cnn_hp):
        def build_cnn(num_layers=3,
                      max_num_channels=64*3,
                      f_size=4):
            x_in = Input(shape=(img_dim, img_dim, 3))
            x = x_in
            for i in range(num_layers + 1):
                num_channels = max_num_channels // 2**(num_layers - i)
                x = Conv2D(num_channels,
                           (5, 5),
                           strides=(2, 2),
                           use_bias=False,
                           padding='same')(x)
                if i > 0:
                    x = BatchNormalization()(x)
                x = LeakyReLU(0.2)(x)
            x = Flatten()(x)
            return Model(x_in, x)
        def build_encoder(cnn_hp)
            cnn_model = build_cnn(**cnn_hp)
            x_in = Input(shape=(img_dim, img_dim, 3))
            x = cnn_model(x_in)
            if kl_weight > -0.5:
                z_mean, z_log_var = Dense(2*z_dim)(x)
                def sampling(args):
                    z_mean, z_log_var = args
                    u = K.random_normal(shape=K.shape(z_mean))
                    return z_mean + K.exp(z_log_var / 2) * u
                z_sample = Lambda(sampling)([z_mean, z_log_var])
                model = Model(x_in,[z_sample, z_mean, z_log_var])
            else:
                z = Dense(z_dim)(x)
                model = Model(x_in, z)
            return model
        def build_dis(cnn_hp):
            cnn_model = build_cnn(**cnn_hp)
            x_in = Input(shape=(z_dim,))
            x = cnn_model(x_in)
            score = Dense(1, use_bias=False)(x)
            model = Model(x_in, score)
            return model
            
        e1 = build_encoder(cnn_hp)
        e2 = e1 if encoder_share else build_encoder(cnn_hp)
        
        # build g_train_model
        x1_in = Input(shape=(img_size, img_size, 3))
        x2_in = Input(shape=(img_size, img_size, 3))
        z1 = e1(x1_in)
        z2 = e2(x2_in)
        g_model = Model([x1_in, x2_in], [z1, z2])
        #build d_model

        


        loss = 0
        if kl_weight > -0.5:
            loss += kl_weight
            
    def build_encoder()
        
    
    def get_img(self, class_p=None, class_y=None):
        x_train, y_train, p_train = self.train_data
        total = x_train.shape[0]
        while True:
            idx = np.random.randint(0, total)
            if class_p is not None and np.argmax(class_p) != np.argmax(p_train[idx]):
                continue
            if class_y is not None and np.argmax(class_y) != np.argmax(y_train[idx]):
                continue
            return x_train[idx]
        return
            
    def sample_all(self, file_path=None):
        g_model, d_model, d_train_model, g_train_model = self.model
        x_train, y_train, p_train = self.train_data
        z_dim = self.z_dim
        num_p = self.num_p
        img_dim = self.img_dim
        num_row = 9
        z = np.random.randn(num_row**2, z_dim)
        p = np.random.choice(num_p, num_row**2)
        p = to_categorical(p)
        predicted = g_model.predict([z, p])
        output = np.zeros((2*num_row*img_dim, num_row*img_dim, 3))
        for i in range(num_row):
            for j in range(num_row):
                output[i*img_dim:(i+1)*img_dim, j*img_dim:(j+1)*img_dim] = self.get_img(class_p=p[i*num_row+j])
                output[(num_row+i)*img_dim:(num_row+i+1)*img_dim, j*img_dim:(j+1)*img_dim] = predicted[i*num_row+j]
        output = (output + 1) / 2 * 255
        output = np.round(output, 0).astype(int)
        if file_path is not None:
            imageio.imwrite(file_path, output)
    def sample_x(self, num):
        x_train,_,_ = self.train_data
        idx = np.random.choice(x_train.shape[0], num, replace=False)
        return x_train[idx]
    def predict(self, x, p=None):
        z_dim = self.z_dim
        z = np.random.randn(x.shape[0], z_dim)
        if p is None:
            num_p = self.num_p
            p = np.random.choice(num_p, x.shape[0])
            p = to_categorical(p)
        g_model, d_model, d_train_model, g_train_model = self.model
        return g_model.predict([z, p])
    def train(self,
              experiment_dir=None,
              total_iter=20000,
              batch_size=128,
              sample_iter=100,
              model_save_iter=1000):
        if experiment_dir is not None:
            sample_dir = experiment_dir / 'sample'
            sample_dir.mkdir(parents=True, exist_ok=True)
            model_dir = experiment_dir / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
        
        z_dim = self.z_dim
        num_p = self.num_p
        g_model, d_model, d_train_model, g_train_model = self.model
        x_train, y_train, p_train = self.train_data
        
        cur_time = time.clock()
        log_iter = 10
        for i in range(total_iter):
            for j in range(2):
                idx = np.random.choice(x_train.shape[0], batch_size, replace=False)
                x_sample = x_train[idx]
                p_real = p_train[idx]
                z_sample = np.random.randn(batch_size, z_dim)
                p_fake = np.random.choice(num_p, batch_size)
                p_fake = to_categorical(p_fake, num_classes=num_p)
                d_loss = d_train_model.train_on_batch([x_sample, z_sample, p_real, p_fake], None)
            for j in range(1):
                idx = np.random.choice(x_train.shape[0], batch_size, replace=False)
                x_sample = x_train[idx]
                p_real = p_train[idx]
                z_sample = np.random.randn(len(x_sample), z_dim)
                p_fake = np.random.choice(num_p, batch_size)
                p_fake = to_categorical(p_fake, num_classes=num_p)
                g_loss = g_train_model.train_on_batch([x_sample, z_sample, p_real, p_fake], None)
            if i % log_iter == 0:
                now = time.clock()
                elapsed = now - cur_time
                cur_time = now
                #import pdb;pdb.set_trace()
                print(f'iter{i} g_loss={g_loss:.2f} d_loss={d_loss:.2f}')
            if i % sample_iter == 0:
                if self.debug:
                    self.sample_all()
                else:
                    self.sample_all(sample_dir / f'{i}.png')
            if model_save_iter > 0 and i % model_save_iter == 0:
                self.save_weights(model_dir/f'{i}')
    def save_weights(self, path):
        combined_path = Path(str(path) + 'combined.h5')
        combined = self.model[-1]
        combined.save_weights(combined_path)
    def load_weights(self, experiment_dir=None, iter_no=None):
        path = experiment_dir / 'models'
        if iter_no is not None:
            path = path/f'{iter_no}'
        combined_path = Path(str(path) + 'combined.h5')
        combined = self.model[-1]
        combined.load_weights(combined_path)
    def evaluate(self, batch_size=None, num_epochs=None):
        x_train, y_train, p_train = self.train_data
        x_test, y_test, p_test = self.test_data

        num_p = self.num_p
        p_fake = np.random.choice(num_p, x_test.shape[0])
        p_fake = to_categorical(p_fake)
        x_test = self.predict(x_test, p=p_fake)
        p_test = p_fake
        resnet = resnet_v1(self.input_shape, self.num_p)
        history = resnet.fit(x_train, p_train, validation_data=(x_test, p_test), \
                             batch_size=batch_size, epochs=num_epochs)

        acc = np.max(history.history['val_acc'])
        print(acc)
    def summary(self):
        for model in self.model:
            print('-'*80)
            model.summary()


def prepareLogger(out_file=None):
    if out_file is not None:
        sys.stdout = TeeLogger(out_file, sys.stdout)
        sys.stderr = TeeLogger(out_file, sys.stderr)        
    
@click.group()
@click.option('--gpu', type=str, default='0')
@click.option('--rand_seed', type=int, default=13141)
def main(gpu, rand_seed):
    seed(rand_seed)
    set_random_seed(rand_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    pass

@main.command()
@click.option('--name', type=str, default='pprl')
@click.option('--epoch', type=int, default=20000)
@click.option('--batch_size', type=int, default=128)
@click.option('--sample_iter', type=int, default=100)
@click.option('--model_save_iter', type=int, default=1000)
@click.option('--recover_dir', type=str)
@click.option('--debug/--no-debug', default=False)
@click.option('--variation/--no-variation', default=True)
@click.option('--test/--no-test', default=True)
def train(name, 
          epoch=20000, 
          debug=False, 
          batch_size=128, 
          sample_iter=100, 
          model_save_iter=1000,
          variation=True,
          recover_dir=None,
          test=True
         ):
    experiment_dir = None
    if not debug:
        experiment_dir = EXPERIMENT_ROOT / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        prepareLogger(experiment_dir/'train_log.txt')
        print('train')
    else:
        model_save_iter = -1
        sample_iter = 1
        epoch = 2
        batch_size = 2
        test=True
        print('debug train')
    loader = load_ferg()
    pprl = Pprl(loader, variation=variation, debug=debug)
    if recover_dir is not None:
        pprl.load_weights(recover_dir)
    pprl.train(experiment_dir=experiment_dir,
                 total_iter=epoch,
                 batch_size=batch_size,
                 sample_iter=sample_iter,
                 model_save_iter=model_save_iter)
    if test:
        pprl.evaluate(batch_size=batch_size)


@main.command()
@click.option('--name', type=str, default='pprl')
@click.option('--iter_no', type=int, default=100)
@click.option('--epoch', type=int, default=20)
@click.option('--batch_size', type=int, default=128)
@click.option('--debug/--no-debug', default=False)
def test(name, iter_no=100, debug=False, batch_size=128, epoch=20):
    experiment_dir = EXPERIMENT_ROOT / name
    prepareLogger()
    print('test')
    loader = load_ferg()
    pprl = Pprl(loader, debug=debug)
    pprl.load_weights(experiment_dir=experiment_dir, iter_no=iter_no)
    pprl.evaluate(batch_size=batch_size, num_epochs=epoch)

@main.command()
def show():
    print('show')
    loader = load_ferg()
    pprl = Pprl(loader)
    pprl.summary()

if __name__ == '__main__':
    main()
