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
a simple sgan-info baseline
"""
def empty_loss(y_true, y_pred):
    return y_pred
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
def show_model(model):
    print('-'*80)
    print(model.summary())
    print(model.metrics_names)
    print('-'*80)
def evaluate_encoder(train_data, test_data, num_classes, batch_size=256, num_epochs=20):
    decoder = build_classifier(num_classes)
    x_train, y_train = train_data
    x_test, y_test = test_data
    decoder.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    history = decoder.fit(x=x_train, y=y_train, epochs=num_epochs,batch_size=batch_size,\
                validation_data=(x_test, y_test),verbose=0)
    return np.max(history.history['val_acc'])

def shuffling(x):
    idxs = K.arange(0, K.shape(x)[0])
    idxs = K.tf.random_shuffle(idxs)
    return K.gather(x, idxs)
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



class AeGan(FergTask):
    """
    expect: able to generate sharp images according to p_fake, but fail to retain other face expression information
    further reading: ALI, cycle gan
    """
    def __init__(self, data_loader, variation=True, z_dim=128, debug=False):
        self.z_dim = z_dim
        self.variation = variation
        super().__init__(data_loader, debug)
    def build_model(self):
        input_shape = self.input_shape
        img_dim = self.img_dim
        z_dim = self.z_dim
        num_p = self.num_p
      
        losses = ['categorical_crossentropy', 'binary_crossentropy']
        def kl_loss(y_true, y_pred):
            z_mean = y_pred[:, 0:z_dim]
            z_log_var = y_pred[:, z_dim:z_dim+z_dim]
            loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return loss
        def q_metric(y_true, y_pred):
            return binary_accuracy(K.round(y_true), y_pred)
        metrics = {'predict_p':'accuracy', 'predict_q': q_metric}
        #optimizer = RMSprop(lr=0.0003, decay=1e-6)
        optimizer = Adam(lr=0.003)
        x_in = Input(input_shape)
        p_in = Input((num_p,))
        encoder = self.build_encoder(input_shape, z_dim)
        if self.variation:
            z, z_mean, z_log_var = encoder(x_in)
            kl_info = Concatenate()([z_mean, z_log_var])
        else:
            z = encoder(x_in)
        decoder = self.build_decoder(z_dim, img_dim, num_p)
        x_hat_fake = decoder([z, p_in])
        classifier = self.build_classifier(input_shape, num_p, z_dim)
        predict_p, predict_q = classifier(x_hat_fake)
        predict_p = Lambda(lambda x:x, name = "predict_p")(predict_p)
        predict_q = Lambda(lambda x:x, name = "predict_q")(predict_q)
        
        classifier.compile(loss=losses,
                           optimizer=optimizer,
                           metrics=metrics)
        classifier.trainable = False
        if not self.variation:
            combined = Model([x_in, p_in], [predict_p, predict_q])
            combined.compile(loss=losses,
                             optimizer=optimizer,
                             metrics=metrics)
        else:
            combined = Model([x_in, p_in], [predict_p, predict_q, kl_info])
            combined.compile(loss=losses+[kl_loss],
                             optimizer=optimizer,
                             loss_weights=[1., 1., 0.03],
                             metrics=metrics)
        model = [encoder, decoder, classifier, combined]
        return model
    def build_encoder(self, input_shape, z_dim):
        x_in = Input(input_shape)
        x = x_in
        field_size = 8
        for i in range(3):
            x = Conv2D(int(z_dim / 2**(2-i)),
                       kernel_size=(field_size, field_size),
                       padding='SAME')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
            x = MaxPooling2D((2, 2))(x)
        x = GlobalMaxPooling2D()(x)
        z_mean = Dense(z_dim)(x)
        if not self.variation:
            return Model(x_in, z_mean)
        z_log_var = Dense(z_dim)(x)
        sample = Lambda(sampling)([z_mean, z_log_var])
        return Model(x_in, [sample, z_mean, z_log_var])
    def build_decoder(self, z_dim, img_dim, num_p):
        k = 8
        units = z_dim
        x_in = Input(shape=(z_dim,))
        p_in =Input(shape=(num_p,))
        h = Concatenate()([x_in, p_in])
        h = Dense(4 * 4 * 128, activation='relu')(h)
        h = Reshape((4, 4, 128))(h)
        # h = LeakyReLU(0.2)(h)
        h = Conv2DTranspose(units, (k, k), strides=(2, 2), padding='same', activation='relu')(h)  # 32*32*64
        # h = Dropout(dropout)(h)
        h = BatchNormalization(momentum=0.8)(h)
        # h = LeakyReLU(0.2)(h)
        # h = UpSampling2D(size=(2, 2))(h)
        h = Conv2DTranspose(units // 2, (k, k), strides=(2, 2), padding='same', activation='relu')(h)  # 64*64*64
        # h = Dropout(dropout)(h)
        h = BatchNormalization(momentum=0.8)(h)
        # h = LeakyReLU(0.2)(h)
        # h = UpSampling2D(size=(2, 2))(h)
        h = Conv2DTranspose(units // 2, (k, k), strides=(2, 2), padding='same', activation='relu')(h)  # 8*6*64
        # h = Dropout(dropout)(h)
        h = BatchNormalization(momentum=0.8)(h)

        h = Conv2DTranspose(3, (k, k), strides=(2, 2), padding='same', activation='tanh')(h)  # 8*6*64
        return Model([x_in, p_in], h)
    def build_classifier(self, input_shape, num_p, z_dim):
        x_in = Input(shape=input_shape)
        x = x_in
        field_size = 8
        for i in range(3):
            x = Conv2D(int(z_dim / 2**(2-i)),
                       kernel_size=(field_size, field_size),
                       padding='SAME')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
            x = MaxPooling2D((2, 2))(x)
        x = GlobalMaxPooling2D()(x)
        predict_p = Dense(num_p, activation='softmax')(x)
        predict_p = Lambda(lambda x:x, name = "predict_p")(predict_p)
        predict_q = Dense(1, activation='sigmoid')(x)
        predict_q = Lambda(lambda x:x, name = "predict_q")(predict_q)
        return Model(x_in, [predict_p, predict_q])
        
    def predict(self, x, p_fake=None):
        encoder, decoder, classifier, combined = self.model
        if self.variation:
            z,_,_ = encoder.predict(x)
        else:
            z = encoder.predict(x)
        batch_size = x.shape[0]
        if p_fake is None:
            p_fake = np.random.randint(self.num_p, size=batch_size)
            p_fake = to_categorical(p_fake, self.num_p)
        x_hat_fake = decoder.predict([z, p_fake])
        return x_hat_fake
    def train(self, experiment_dir=None, total_iter=20000, batch_size=128, sample_iter=100, model_save_iter=1000):
        sample_dir = experiment_dir / 'sample'
        sample_dir.mkdir(parents=True, exist_ok=True)
        model_dir = experiment_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        z_dim = self.z_dim
        encoder, decoder, classifier, combined = self.model
        x_train, y_train, p_train = self.train_data
        
        cur_time = time.clock()
        log_iter = 10
        for i in range(total_iter):
            idx1 = np.random.choice(x_train.shape[0], batch_size, replace=False)
            idx2 = np.random.choice(x_train.shape[0], batch_size, replace=False)
            p_fake = np.random.randint(self.num_p, size=batch_size)
            p_fake = to_categorical(p_fake, self.num_p)
            x_fake = self.predict(x_train[idx1], p_fake=p_fake)
            q_valid = np.random.uniform(0.9, 1.0, size=(batch_size, 1))
            q_fake = np.random.uniform(0.0, 0.1, size=(batch_size, 1))
            
            d_loss_real = classifier.train_on_batch(x_train[idx2], [p_train[idx2], q_valid])
            d_loss_fake = classifier.train_on_batch(x_fake, [p_fake, q_fake])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            for j in range(1):
                p_fake = np.random.randint(self.num_p, size=batch_size)
                p_fake = to_categorical(p_fake, self.num_p)
                y_target = [p_fake, q_valid]
                if self.variation:
                    y_target = [p_fake, q_valid, np.ones((batch_size, z_dim*2))]
                g_loss = combined.train_on_batch([x_train[idx1], p_fake], y_target)
            if i % log_iter == 0:
                now = time.clock()
                elapsed = now - cur_time
                cur_time = now
                #import pdb;pdb.set_trace()
                print(f'iter: {i}\nd: loss_q={d_loss[2]:.4f} acc_p={d_loss[-2]*100:.2f} acc_q={d_loss[-1]*100:.2f}')
                if self.variation:
                    print(f'g: loss= {g_loss[0]:.4f} kl_loss={g_loss[2]:.4f} acc_q={g_loss[-1]*100:.2f}, elapsed: {elapsed:.2f}')
                else:
                    print(f'g: loss= {g_loss[0]:.4f} acc_p={g_loss[-2]*100:.2f} acc_q={g_loss[-1]*100:.2f}, elapsed: {elapsed:.2f}')
            if i % sample_iter == 0:
                self.sample_all(sample_dir / f'{i}.png')
            if model_save_iter >0 and i % model_save_iter == 0:
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
    def summary(self):
        encoder, decoder, classifier, combined = self.model
        print('encoder')
        encoder.summary()
        print('decoder')
        decoder.summary()
        print('classifier')
        classifier.summary()
        print('combined')
        combined.summary()

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
@click.option('--name', type=str, default='0')
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
    name = 'ae_gan' + name
    experiment_dir = EXPERIMENT_ROOT / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    prepareLogger(experiment_dir/'train_log.txt')
    print('train')
    print(f'{name} aegan without reconstruction loss')
    loader = load_ferg()
    ae_gan = AeGan(loader, variation=variation, debug=debug)
    if recover_dir is not None:
        ae_gan.load_weights(recover_dir)
    ae_gan.train(experiment_dir, total_iter=epoch, batch_size=batch_size, sample_iter=sample_iter, model_save_iter=model_save_iter)
    if test:
        acc_y_on_x = ae_gan.evaluate_y_on_x()
        acc_p_on_x = ae_gan.evaluate_p_on_x()
        print(f'acc_y_on_x = {acc_y_on_x}, acc_p_on_x={acc_p_on_x}')


@main.command()
@click.option('--name', type=str, default='0')
@click.option('--iter_no', type=int, default=100)
@click.option('--num_epochs', type=int, default=20)
@click.option('--debug/--no-debug', default=False)
def test(name, iter_no=100, debug=False, num_epochs=20):
    name = 'ae_gan' + name
    experiment_dir = EXPERIMENT_ROOT / name
    prepareLogger()
    print('test')
    loader = load_ferg()
    ae_gan = AeGan(loader, debug=debug)
    ae_gan.load_weights(experiment_dir=experiment_dir, iter_no=iter_no)
    acc_y_on_x = ae_gan.evaluate_y_on_x(num_epochs=num_epochs)
    acc_p_on_x = ae_gan.evaluate_p_on_x(num_epochs=num_epochs)
    print(f'acc_y_on_x = {acc_y_on_x}, acc_p_on_x={acc_p_on_x}')

@main.command()
def show():
    print('show')
    loader = load_ferg()
    ae_gan = AeGan(loader)
    ae_gan.summary()

if __name__ == '__main__':
    main()