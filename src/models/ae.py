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

from pathlib import Path
import numpy as np
from src.data.dataset import load_ferg
from src.evaluation.resnet import resnet_v1
from src import PROJECT_ROOT
import imageio
import os
from src.util.callbacks import Evaluate
from src.models.task import FergTask
import click

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
def rec_loss_func(args):
    y_true, y_pred = args
    return K.mean(K.square(y_pred - y_true))
def categorical_loss_func(args):
    y_true, y_pred = args
    return categorical_crossentropy(y_true, y_pred)


class Ae(FergTask):
    def __init__(self, data_loader, z_dim=128, debug=False):
        self.z_dim = z_dim
        super().__init__(data_loader, debug)
    def build_model(self):
        input_shape = self.input_shape
        img_dim = self.img_dim
        z_dim = self.z_dim
        encoder = self.build_encoder(input_shape, z_dim)
        decoder = self.build_decoder(z_dim, img_dim)
        ae = self.build_ae(input_shape, encoder, decoder)
        model = [encoder, decoder, ae]
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
        return Model(x_in, z_mean)
    def build_decoder(self, z_dim, img_dim):
        k = 8
        units = z_dim
        x = Input(shape=(z_dim,))
        h = x
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
        return Model(x, h)
    def build_ae(self, input_shape, encoder, decoder):
        x_in = Input(shape=input_shape)
        z = encoder(x_in)
        x_rec = decoder(z)
        rec_loss = Lambda(rec_loss_func)([x_in, x_rec])
        return Model(x_in, rec_loss)
    def predict(self, x):
        encoder, decoder, ae = self.model
        z = encoder.predict(x)
        rec_x = decoder.predict(z)
        return rec_x
    def train(self, sample_dir=None, model_path=None, num_epochs=20, batch_size=128):
        encoder, decoder, ae = self.model
        ae.compile(optimizer=Adam(1e-4), loss=empty_loss)
        x_train, y_train, p_train = self.train_data
        x_test, y_test, p_test = self.test_data
        evaluator = Evaluate(self, sample_dir=sample_dir, model_path=model_path)
        ae.fit(x_train, x_train, validation_data=(x_test, x_test), \
               batch_size=batch_size, epochs=num_epochs, callbacks=[evaluator])
@click.command()
@click.option('--gpu', type=str)
@click.option('--epoch', type=int, default=20)
@click.option('--debug/--no-debug', default=True)
def main(gpu, epoch, debug):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    loader = load_ferg()
    ae = Ae(loader, debug=debug)
    ae.train(num_epochs=epoch)
    acc_y_on_x = ae.evaluate_y_on_x()
    acc_p_on_x = ae.evaluate_p_on_x()
    print(f'acc_y_on_x = {acc_y_on_x}, acc_p_on_x={acc_p_on_x}')

if __name__ == '__main__':
    main()