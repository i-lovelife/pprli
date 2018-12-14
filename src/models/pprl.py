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
from src.data.dataset import Dataset
from src.evaluation import Evaluator
from src.models.privater import Privater
import click
"""
pprl model
"""
class Pprl(Privator):
    def __init__(self,
                 dataset='ferg',
                 img_dim=64,
                 z_dim=128,
                 kl_weight=-1.,
                 rec_weight=-1.,
                 encoder_share=False,
                 cls_y_weight=-1.,
                 cnn_hp={},
                 eva_hp={}):
        self.dataset = Dataset.by_name(dataset)
        self.eva_hp = eva_hp
        def build_cnn
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
        def kl_loss(z_mean, z_log_var):
            return - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
            
        e1 = build_encoder(cnn_hp)
        e2 = e1 if encoder_share else build_encoder(cnn_hp)
        d = build_dis(cnn_hp)
        
        # build d_train_model
        e1.trainable = False
        e2.trainable = False
        d.trainable = True
        x1_in = Input(shape=(img_dim, img_dim, 3))
        x2_in = Input(shape=(img_dim, img_dim, 3))
        if kl_weights<0:
            z1 = e1(x1_in)
            z2 = e2(x2_in)
        else:
            z1, z1_mean, z1_log_var = e1(x1_in)
            z2, z2_mean, z2_log_var = e2(x2_in)
        score1 = d(z1)
        score2 = d(z2)
        d_train_model = Model([x1_in, x2_in], [score1, score2])
        d_loss = score1 - score2
        d_norm = 10 * K.mean(K.abs(x_real - x_fake), axis=[1, 2, 3])
        d_loss = K.mean(- d_loss + 0.5 * d_loss**2 / d_norm)
        if kl_weights<0:
            loss = d_loss
        else:
            loss = d_loss + kl_weights* kl_loss(z_mean, z_log_var)
        d_train_model.add_loss(loss)
        d_train_model.compile(optimizer=Adam(2e-4, 0.5))

        #build g_train_model
        e1.trainable = True
        e2.trainable = True
        d.trainable = False
        x1_in = Input(shape=(img_dim, img_dim, 3))
        x2_in = Input(shape=(img_dim, img_dim, 3))
        if kl_weights<0:
            z1 = e1(x1_in)
            z2 = e2(x2_in)
        else:
            z1, z1_mean, z1_log_var = e1(x1_in)
            z2, z2_mean, z2_log_var = e2(x2_in)
        score1 = d(z1)
        score2 = d(z2)
        g_train_model = Model([x1_in, x2_in], [score1, score2])
        g_loss = K.mean(score1 - score2)
        if kl_weights<0:
            loss = g_loss
        else:
            loss = g_loss + kl_weights* kl_loss(z_mean, z_log_var)
        g_train_model.add_loss(loss)
        g_train_model.compile(optimizer=Adam(2e-4, 0.5))

        self.e1 = e1
        self.e2 = e2
        self.d = d
        self.d_train_model = d_train_model
        self.g_train_model = g_train_model
    
    def predict(self, x1, x2):
        return self.g1.predict(x1), self.g2.predict(x2)
    def train(self,
              experiment_dir=None,
              total_iter=20000,
              batch_size=128,
              evaluate_iter=500):
        if experiment_dir is not None:
            sample_dir = experiment_dir / 'sample'
            sample_dir.mkdir(parents=True, exist_ok=True)
            model_dir = experiment_dir / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = self.dataset
        cur_time = time.clock()
        log_iter = 10
        for i in range(total_iter):
            for j in range(2):
                x1_batch, x2_batch = dataset.gen_batch(batch_size=batch_size)
                d_loss = d_train_model.train_on_batch([x1_batch, x2_batch], None)
            for j in range(1):
                x1_batch, x2_batch = dataset.gen_batch(batch_size=batch_size)
                g_loss = g_train_model.train_on_batch([x1_batch, x2_batch], None)
            if i % log_iter == 0:
                now = time.clock()
                elapsed = now - cur_time
                cur_time = now
                #import pdb;pdb.set_trace()
                print(f'iter{i} g_loss={g_loss:.2f} d_loss={d_loss:.2f}')
            if i % evaluate_iter == 0:
                p_acc, u_acc = self.evaluate()
                self.save_weights_iter(model_dir, i, p_acc, u_acc)

    def get_name(self):
        return type(self).__name__

    def save_weights(self, path):
        self.g_train_model.save_weights(path)

    def save_weights_iter(self, model_dir, iter_no, p_acc, u_acc):
        name = self.get_name()
        path = model_dir / f'{name}_{iter_no}_p{p_acc*100:.0f}_u{u_acc*100:.0f}.h5'
        self.save_weights(path)

    def load_weights(self, model_path):
        self.g_train_model.load_weights(model_path)

    def evaluate(self):
        eva_hp = self.eva_hp
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
