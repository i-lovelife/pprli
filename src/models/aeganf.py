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
import imageio
import os
from src.util.callbacks import Evaluate
from src.models.task import FergTask
import click

logger = logging.getLogger('ae_ganf')
logger.setLevel(logging.INFO)
"""
Test whether reconstruction influence result
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
def kl_loss_func(args):
    z_mean, z_log_var = args
    loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return loss

def spectral_norm(w, r=5):
    w_shape = K.int_shape(w)
    in_dim = np.prod(w_shape[:-1]).astype(int)
    out_dim = w_shape[-1]
    w = K.reshape(w, (in_dim, out_dim))
    u = K.ones((1, in_dim))
    for i in range(r):
        v = K.l2_normalize(K.dot(u, w))
        u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))


def spectral_normalization(w):
    return w / spectral_norm(w)


class AeGanF(FergTask):
    def __init__(self, data_loader, z_dim=128, debug=False):
        self.z_dim = z_dim
        super().__init__(data_loader, debug)
    def build_model(self):
        input_shape = self.input_shape
        img_dim = self.img_dim
        z_dim = self.z_dim
        num_p = self.num_p
        encoder = self.build_encoder(input_shape, z_dim)
        decoder = self.build_decoder(z_dim, img_dim, num_p)
        classifier = self.build_classifier(input_shape, num_p, z_dim)
        generator = self.build_generator(input_shape, num_p, encoder, decoder, classifier)
        discriminator = self.build_discriminator(input_shape, num_p, classifier)
        model = [encoder, decoder, classifier, generator, discriminator]
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
        x = Dense(z_dim)(x)
        return Model(x_in, x)
        #z_mean = Dense(z_dim)(x)
        #z_log_var = Dense(z_dim)(x)
        #sample = Lambda(sampling)([z_mean, z_log_var])
        #return Model(x_in, [sample, z_mean, z_log_var])
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
                       kernel_constraint=spectral_normalization,
                       padding='SAME')(x)
            x = BatchNormalization(gamma_constraint=spectral_normalization)(x)
            x = LeakyReLU(0.2)(x)
            x = MaxPooling2D((2, 2))(x)
        x = GlobalMaxPooling2D()(x)
        predict_p = Dense(num_p, activation='softmax')(x)
        predict_q = Dense(1, use_bias=False, kernel_constraint=spectral_normalization)(x)
        return Model(x_in, [predict_p, predict_q])
    def build_generator(self, input_shape, num_p, encoder, decoder, classifier):
        x_in = Input(shape=input_shape)
        p_in_real = Input(shape=(num_p,))
        p_in_fake = Input(shape=(num_p,))
        x = x_in
        x = encoder(x)
        x_hat_fake = decoder([x, p_in_fake])
        x_hat_real = decoder([x, p_in_real])
        def mse_loss_func(args):
            y_true, y_pred = args
            return K.mean(K.square(y_pred - y_true))
        fake_p, fake_q = classifier(x_hat_fake)
        def p_loss_func(args):
            y_true, y_pred = args
            return K.mean(categorical_crossentropy(y_true, y_pred))
        p_loss = Lambda(p_loss_func)([p_in_fake, fake_p])
        def q_loss_func(args):
            fake_q = args
            return K.mean(- fake_q)
        q_loss = Lambda(q_loss_func)(fake_q)
        def add_func(args):
            p_loss, q_loss= args
            return p_loss + q_loss
        loss = Lambda(add_func)([p_loss, q_loss])
        model = Model([x_in, p_in_real, p_in_fake], loss)
        classifier.trainable = False
        model.compile(RMSprop(lr=0.0003, decay=1e-6), loss=empty_loss)
        return model
    def build_discriminator(self, input_shape, num_p, classifier):
        x_fake = Input(shape=input_shape)
        p_fake = Input(shape=(num_p,))
        x_sample = Input(shape=input_shape)
        p_sample = Input(shape=(num_p,))
        
        p_fake_pre, q_fake_pre = classifier(x_fake)
        p_sample_pre, q_sample_pre = classifier(x_sample)
        #p_loss
        def p_loss_func(args):
            y_true, y_pred = args
            return K.mean(categorical_crossentropy(y_true, y_pred))
        p_loss_layer = Lambda(p_loss_func)
        p_loss_fake = p_loss_layer([p_fake, p_fake_pre])
        p_loss_sample = p_loss_layer([p_sample, p_sample_pre])
        def average_loss_func(args):
            p_loss_fake, p_loss_sample = args
            return (p_loss_fake+p_loss_sample)/2
        p_loss = Lambda(average_loss_func)([p_loss_fake, p_loss_sample])
        #q_loss
        def q_loss_func(args):
            q_sample_pre, q_fake_pre = args
            return K.mean(q_fake_pre - q_sample_pre)
        q_loss = Lambda(q_loss_func)([q_sample_pre, q_fake_pre])
        def add_func(args):
            p_loss, q_loss = args
            return p_loss+q_loss
        loss = Lambda(add_func)([p_loss, q_loss])
        model = Model([x_fake, p_fake, x_sample, p_sample], loss)
        classifier.trainable = True
        model.compile(RMSprop(lr=0.0003, decay=1e-6), loss=empty_loss)
        return model
        
    def predict(self, x, p_fake=None):
        encoder, decoder, classifier, generator, discriminator = self.model
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
        encoder, decoder, classifier, generator, discriminator = self.model
        x_train, y_train, p_train = self.train_data
        
        cur_time = time.clock()
        log_iter = 10
        for i in range(total_iter):
            idx1 = np.random.choice(x_train.shape[0], batch_size, replace=False)
            idx2 = np.random.choice(x_train.shape[0], batch_size, replace=False)
            p_fake = np.random.randint(self.num_p, size=batch_size)
            p_fake = to_categorical(p_fake, self.num_p)
            x_fake = self.predict(x_train[idx1], p_fake=p_fake)
            for j in range(2):
                d_loss = discriminator.train_on_batch([x_fake, p_fake, x_train[idx2], p_train[idx2]], p_train[idx1])
            for j in range(1):
                p_fake = np.random.randint(self.num_p, size=batch_size)
                p_fake= to_categorical(p_fake, self.num_p)
                g_loss = generator.train_on_batch([x_train[idx1], p_train[idx1], p_fake], p_train[idx1])
            if i % log_iter == 0:
                now = time.clock()
                elapsed = now - cur_time
                cur_time = now
                logger.info(f'iter: {i}, d_loss: {d_loss:.6f}, g_loss: {g_loss:.6f} elapsed: {elapsed:.2f}')
            if i % sample_iter == 0:
                self.sample_all(sample_dir / f'{i}.png')
            if model_save_iter >0 and i % model_save_iter == 0:
                self.save_weights(model_dir/f'{i}')
    def save_weights(self, path):
        generator_path = Path(str(path) + 'generator.h5')
        discriminator_path = Path(str(path) + 'discriminator.h5')
        _,_,_, generator, discriminator = self.model
        generator.save_weights(generator_path)
        discriminator.save_weights(discriminator_path)
    def load_weights(self, experiment_dir=None, iter_no=None):
        path = experiment_dir / 'models'
        if iter_no is not None:
            path = path/f'{iter_no}'
        generator_path = Path(str(path) + 'generator.h5')
        discriminator_path = Path(str(path) + 'discriminator.h5')
        _,_,_, generator, discriminator = self.model
        generator.load_weights(generator_path)
        discriminator.load_weights(discriminator_path)
    def summary(self):
        encoder, decoder, classifier, generator, discriminator = self.model
        print('encoder')
        encoder.summary()
        print('decoder')
        decoder.summary()
        print('classifier')
        classifier.summary()
        print('generator')
        generator.summary()
        print('discriminator')
        discriminator.summary()

def prepareLogger(out_file=None):
    if out_file is not None:
        print(out_file)
        file_handler = logging.FileHandler(out_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)        
    
@click.group()
@click.option('--gpu', type=str, default='3')
@click.option('--rand_seed', type=int, default=13141)
def main(gpu, rand_seed):
    seed(rand_seed)
    set_random_seed(rand_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    pass

@main.command()
@click.option('--name', type=str, default='ae_ganf0')
@click.option('--epoch', type=int, default=20000)
@click.option('--batch_size', type=int, default=128)
@click.option('--sample_iter', type=int, default=100)
@click.option('--model_save_iter', type=int, default=1000)
@click.option('--recover_dir', type=str)
@click.option('--debug/--no-debug', default=False)
@click.option('--test/--no-test', default=True)
def train(name, 
          epoch=20000, 
          debug=False, 
          batch_size=128, 
          sample_iter=100, 
          model_save_iter=1000, 
          recover_dir=None,
          test=True
         ):
    experiment_dir = EXPERIMENT_ROOT / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    prepareLogger(experiment_dir/'train_log.txt')
    logger.info('train')
    logger.info(f'{name} aegan without reconstruction loss')
    loader = load_ferg()
    ae_ganf = AeGanF(loader, debug=debug)
    if recover_dir is not None:
        ae_ganf.load_weights(recover_dir)
    ae_ganf.train(experiment_dir, total_iter=epoch, batch_size=batch_size, sample_iter=sample_iter, model_save_iter=model_save_iter)
    if test:
        acc_y_on_x = ae_ganf.evaluate_y_on_x()
        acc_p_on_x = ae_ganf.evaluate_p_on_x()
        logger.info(f'acc_y_on_x = {acc_y_on_x}, acc_p_on_x={acc_p_on_x}')


@main.command()
@click.option('--name', type=str, default='ae_ganf0')
@click.option('--iter_no', type=int, default=100)
@click.option('--num_epochs', type=int, default=20)
@click.option('--debug/--no-debug', default=False)
def test(name, iter_no=100, debug=False, num_epochs=20):
    experiment_dir = EXPERIMENT_ROOT / name
    prepareLogger()
    logger.info('test')
    loader = load_ferg()
    ae_ganf = AeGanF(loader, debug=debug)
    ae_ganf.load_weights(experiment_dir=experiment_dir, iter_no=iter_no)
    acc_y_on_x = ae_ganf.evaluate_y_on_x(num_epochs=num_epochs)
    acc_p_on_x = ae_ganf.evaluate_p_on_x(num_epochs=num_epochs)
    logger.info(f'acc_y_on_x = {acc_y_on_x}, acc_p_on_x={acc_p_on_x}')

@main.command()
def show():
    logger.info('show')
    loader = load_ferg()
    ae_ganf = AeGanF(loader)
    ae_ganf.summary()

if __name__ == '__main__':
    main()