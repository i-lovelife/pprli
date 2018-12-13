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

class Task:
    def save_weights(self, path):
        self.model[-1].save_weights(path)
    def load_weights(self, path):
        self.model[-1].load_weights(path)
    def predict(self, x):
        raise NotImplementedError
    def build_model(self):
        raise NotImplementedError
    def predict_single(self, x):
        x = np.expand_dims(x, axis=0)
        rec_x = self.predict(x)[0]
        return rec_x
class FergTask(Task):
    def __init__(self, data_loader, debug=False):
        #get data
        train_data, test_data, num_y, num_p, num_train, num_test, img_dim, input_shape = self.load_data(data_loader, debug)
        #add config      
        self.img_dim = img_dim
        self.input_shape = input_shape
        self.num_y = num_y
        self.num_p = num_p
        self.num_train = num_train
        self.num_test = num_test
        self.train_data = train_data
        self.test_data = test_data
        #build model
        self.model = self.build_model()
    @staticmethod
    def transform_data(data, num_y, num_p):
        x, y, p= data
        x = (x-127.5)/127.5
        y = to_categorical(y, num_y)
        p = to_categorical(p, num_p)
        return (x, y, p)
    @staticmethod
    def recover_data(data):
        x, y, p = data
        x = (x + 1) / 2 * 255
        x = np.clip(x, 0, 255)
        x = x.astype(np.uint8)
        y = np.argmax(y, axis=-1)
        p = np.argmax(p, axis=-1)
        return x, y, p
    def sample_all(self, file_path):
        x_train, y_train, p_train = self.train_data
        num_y = self.num_y
        num_p = self.num_p
        num_data = self.num_train
        img_dim = self.img_dim
        output = np.zeros((2*num_p*img_dim, num_y*img_dim, 3))
        for i in range(num_p):
            for j in range(num_y):
                for idx in range(num_data):
                    if (np.argmax(p_train[idx]) == i) and (np.argmax(y_train[idx]) == j):
                        x, y, p = x_train[idx], y_train[idx], p_train[idx]
                        x_fake = self.predict_single(x)
                        output[i*img_dim:(i+1)*img_dim, j*img_dim:(j+1)*img_dim,:] = type(self).recover_data((x, y, p))[0]
                        output[i*img_dim+num_p*img_dim:(i+1)*img_dim+num_p*img_dim, \
                               j*img_dim:(j+1)*img_dim,:] = type(self).recover_data((x_fake, y, p))[0]
                        break
        imageio.imwrite(file_path, output)
    def evaluate_y_on_x(self, num_epochs=20, batch_size=128):
        x_train, y_train, p_train = self.train_data
        x_test, y_test, p_test = self.test_data
        x_train = self.predict(x_train)
        x_test = self.predict(x_test)
        resnet = resnet_v1(self.input_shape, self.num_y)
        history = resnet.fit(x_train, y_train, validation_data=(x_test, y_test), \
                             batch_size=batch_size, epochs=num_epochs)
        acc = np.max(history.history['val_acc'])
        return acc
    def evaluate_p_on_x(self, num_epochs=20, batch_size=128):
        x_train, y_train, p_train = self.train_data
        x_test, y_test, p_test = self.test_data
        x_train = self.predict(x_train)
        x_test = self.predict(x_test)
        resnet = resnet_v1(self.input_shape, self.num_p)
        history = resnet.fit(x_train, p_train, validation_data=(x_test, p_test), \
                             batch_size=batch_size, epochs=num_epochs)
        acc = np.max(history.history['val_acc'])
        return acc
    def load_data(self, data_loader, debug):
        if debug:
            train_data, test_data = data_loader.load_data(max_train=500, max_test=500)
        else:
            train_data, test_data = data_loader.load_data()
        num_y, num_p = data_loader.get_num_classes()
        num_train = train_data[0].shape[0]
        num_test = test_data[0].shape[0]
        img_dim = test_data[0].shape[1]
        input_shape = (img_dim, img_dim, 3)
        train_data = type(self).transform_data(train_data, num_y, num_p)
        test_data = type(self).transform_data(test_data, num_y, num_p)
        return train_data, test_data, num_y, num_p, num_train, num_test, img_dim, input_shape

class Ferg2PeopleTask(Task):
    def __init__(self, data_loader, debug=False):
        #get data
        train_data, test_data, num_y, num_p, num_train, num_test, img_dim, input_shape = self.load_data(data_loader, debug)
        #add config      
        self.img_dim = img_dim
        self.input_shape = input_shape
        self.num_y = num_y
        self.num_p = num_p
        self.num_train = num_train
        self.num_test = num_test
        self.train_data = train_data
        self.test_data = test_data
        #build model
        self.model = self.build_model()