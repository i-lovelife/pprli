import numpy as np
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from src.util.worker import Worker

class Privater(Worker):
    _default_type='vae'
    def predict(self, data):
        raise NotImplementedError
    
    def get_input(self, data):
        """return (x, y)
        """
        raise NotImplementedError
    
    def save_weights(self, path):
        self.train_model.save_weights(path)
        
    def load_weights(self, path):
        self.train_model.save_weights(path)
        
    def summary(self):
        self.train_model.summary()
        print(self.train_model.metrics_names)

def build_encoder(img_dim=64, 
                  z_dim=128, 
                  num_conv=5, 
                  use_max_pooling=True, 
                  drop_out=-1,
                  use_gauss_prior=False
                 ):
    x_in = Input(shape=(img_dim, img_dim, 3))
    x = x_in
    for i in range(num_conv):
        x = Conv2D(z_dim // 2**(num_conv-i),
                   kernel_size=(3,3),
                   padding='SAME')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        if drop_out > 0:
            x = Dropout(dropout)(x)
        if use_max_pooling:
            x = MaxPooling2D((2, 2))(x)

    if use_max_pooling:
        x = GlobalMaxPooling2D()(x)
    else:
        x = Flatten()(x)

    if use_gauss_prior:
        z_mean = Dense(z_dim)(x)
        z_log_var = Dense(z_dim)(x)
        return Model(x_in, [z_mean, z_log_var])
    z = Dense(z_dim)(x)
    return Model(x_in, z)

def build_classifier(img_dim=64, 
                     num_classes=6,
                     z_dim=128,
                     **args):
    args['use_gauss_prior'] = False
    encoder = build_encoder(z_dim=z_dim, img_dim=img_dim, **args)
    x_in = Input(shape=(img_dim, img_dim, 3))
    x = encoder(x_in)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(x_in, x)

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

def gauss_sampling(args):
    z_mean, z_log_var = args
    u = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * u

def gauss_loss_func(args):
    z_mean, z_log_var = args
    return - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
def identity_loss(y_true, y_pred):
    return y_pred

@Privater.register('vae')
class Vae(Privater):
    def __init__(self,
                 img_dim=64,
                 z_dim=128,
                 loss_weights={'prior':1, 'rec_x': 64*64/10}
                 ):
        encoder = build_encoder(img_dim=img_dim,
                                z_dim=z_dim,
                                use_max_pooling=True, 
                                drop_out=-1,
                                use_gauss_prior=True)
        x_in = Input(shape=(img_dim, img_dim, 3))
        x = x_in
        z_mean, z_log_var = encoder(x)
        z = Lambda(gauss_sampling)([z_mean, z_log_var])
        gauss_loss = Lambda(gauss_loss_func, name='prior')([z_mean, z_log_var])
        decoder = build_decoder(img_dim=img_dim,
                                z_dim=z_dim)
        rec_x = decoder(z)
        rec_x = Lambda(lambda x: x, name="rec_x")(rec_x)
        train_model = Model(x_in, [rec_x, gauss_loss])
        
        train_model.compile(optimizer=Adam(1e-3),
                            loss={'prior': identity_loss, 'rec_x': 'mean_squared_error'},
                            loss_weights=loss_weights
                            )
        self.encoder = encoder
        self.decoder = decoder
        self.train_model = train_model
    
    def get_input(self, data):
        x, y, p = data['x'], data['y'], data['p']
        return (x, {'prior':x, 'rec_x':x})
    def predict(self, data):
        x, y, p = data['x'], data['y'], data['p']
        x, _ = self.encoder.predict(x, batch_size=32)
        return {'x': x, 'y': y, 'p': p}
        
@Privater.register('vae_gan')
class VaeGan(Privater):
    def __init__(self,
                 img_dim=64,
                 z_dim=128,
                 p_dim=6,
                 rec_z_loss_weight=10,
                 real_rec_x_weight=10,
                 real_pred_p_weight=0.5,
                 fake_pred_p_weight=0.5
                 ):
        encoder = build_encoder(img_dim=img_dim,
                                z_dim=z_dim,
                                use_max_pooling=True, 
                                drop_out=-1,
                                use_gauss_prior=True)
        x_in = Input(shape=(img_dim, img_dim, 3))
        real_p_in = Input(shape=(p_dim,))
        fake_p_in = Input(shape=(p_dim,))
        x = x_in
        z_mean, z_log_var = encoder(x)
        z = Lambda(gauss_sampling)([z_mean, z_log_var])
        gauss_loss = Lambda(gauss_loss_func, name='gauss_loss')([z_mean, z_log_var])
        
        real_z = Concatenate()([z, real_p_in])
        fake_z = Concatenate()([z, fake_p_in])
        decoder = build_decoder(img_dim=img_dim,
                                z_dim=z_dim+p_dim)
        real_rec_x = decoder(real_z)
        real_rec_x = Lambda(lambda x: x, name="real_rec_x")(real_rec_x)
        
        fake_rec_x = decoder(fake_z)
        fake_z_mean, fake_z_log_var = encoder(fake_rec_x)
        #calculate KL divergence between N(z_mean, z_log_var) and N(fake_z_mean, fake_z_log_var)
        rec_z_loss = Lambda(gauss_loss_func, name='rec_z_loss')([fake_z_mean, fake_z_log_var])
        
        classifier = build_classifier(img_dim=img_dim,
                                      z_dim=z_dim)
        real_pred_p = classifier(x_in)
        real_pred_p = Lambda(lambda x: x, name='real_pred_p')(real_pred_p)
        fake_pred_p = classifier(fake_rec_x)
        fake_pred_p = Lambda(lambda x: x, name='fake_pred_p')(fake_pred_p)
        
        train_model = Model([x_in, real_p_in, fake_p_in], 
                            [gauss_loss, real_rec_x, rec_z_loss, real_pred_p, fake_pred_p])
        
        train_model.compile(optimizer=Adam(1e-3),
                            loss={'gauss_loss': identity_loss, 
                                  'rec_z_loss':identity_loss,
                                  'real_rec_x': 'mean_squared_error',
                                  'real_pred_p': 'categorical_crossentropy',
                                  'fake_pred_p': 'categorical_crossentropy'},
                            loss_weights={'gauss_loss':1,
                                          'rec_z_loss':rec_z_loss_weight,
                                          'real_rec_x':real_rec_x_weight,
                                          'real_pred_p':real_pred_p_weight,
                                          'fake_pred_p':fake_pred_p_weight},
                            metrics={'real_pred_p': 'acc',
                                     'fake_pred_p': 'acc'}
                            )
        self.encoder = encoder
        self.decoder = decoder
        self.train_model = train_model
    
    def gen_random_p(self, shape):
        p = np.random.randint(shape[1], size=shape[0])
        return to_categorical(p, num_classes=shape[1])
    def get_input(self, data):
        x, y, p = data['x'], data['y'], data['p']
        fake_p = self.gen_random_p(p.shape)
        return ([x, p, fake_p], {'gauss_loss':x,
                                 'rec_z_loss':x,
                                 'real_rec_x':x,
                                 'real_pred_p':p,
                                 'fake_pred_p':fake_p
                                })
    def predict(self, data):
        x, y, p = data['x'], data['y'], data['p']
        x, _ = self.encoder.predict(x, batch_size=32)
        return {'x': x, 'y': y, 'p': p}
    
@Privater.register('adae')
class Adae(Privater):
    def __init__(self,
                 img_dim=64,
                 z_dim=128,
                 g_train_weights={'p':-1, 'rec_x':50},
                 d_train_weights={'p':1}
                 ):
        encoder = build_encoder(img_dim=img_dim,
                                z_dim=z_dim,
                                use_max_pooling=True, 
                                drop_out=-1, 
                                output_feature_map=-1)
        decoder = build_decoder(img_dim=img_dim,
                                z_dim=z_dim)
            
        def build_discrim():
            x_in = Input(shape=(z_dim,))
            x = x_in
            x = Dense(z_dim, activation='relu')(x)
            x = Dense(6, activation='softmax')(x)
            return Model(x_in, x)
        
        classifier = build_discrim()
        x_in = Input(shape=(img_dim, img_dim, 3))
        z = encoder(x_in)
        rec_x = decoder(z)
        rec_x = Lambda(lambda x: x, name="rec_x")(rec_x)
        p_pred = classifier(z)
        p_pred = Lambda(lambda x: x, name="p")(p_pred)
        
        #build g_train_model
        g_train_model = Model(x_in, [rec_x, p_pred])
        encoder.trainable = True
        decoder.trainable = True
        classifier.trainable = False
        g_train_model.compile(optimizer=Adam(1e-3),
                              loss={'p': 'categorical_crossentropy', 'rec_x': 'mean_squared_error'},
                              loss_weights=g_train_weights,
                              metrics={'p': 'acc'})
        
        #build d_train_model
        d_train_model = Model(x_in, p_pred)
        encoder.trainable = False
        decoder.trainable = False
        classifier.trainable = True
        d_train_model.compile(optimizer=Adam(1e-3),
                              loss={'p': 'categorical_crossentropy'},
                              loss_weights=d_train_weights,
                              metrics={'p': 'acc'})
        
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.g_train_model = g_train_model
        self.d_train_model = d_train_model
        self.train_model = g_train_model
        
    def predict(self, data):
        x, y, p = data['x'], data['y'], data['p']
        x = self.encoder.predict(x, batch_size=32)
        return {'x': x, 'y': y, 'p': p}
    
    def get_input(self, data):
        pass
    def get_input_d(self, batch_data):
        x = batch_data['x']
        p = batch_data['p']
        return (x, {'p': p})
        return loss
    def get_input_g(self, batch_data):
        x = batch_data['x']
        p = batch_data['p']
        return (x, {'p':p, 'rec_x':x})
