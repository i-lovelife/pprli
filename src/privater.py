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

def build_discriminator(img_dim=64,
                        num_classes=-1,
                        z_dim=128,
                        **args):
    args['use_gauss_prior'] = False
    encoder = build_encoder(z_dim=z_dim, img_dim=img_dim, **args)
    x_in = Input(shape=(img_dim, img_dim, 3))
    x = encoder(x_in)
    judge = Dense(1, use_bias=False)(x)
    if num_classes > 0:
        classify = Dense(num_classes, activation='softmax')(x)
        return Model(x_in, [judge, classify])
    return Model(x_in, judge)

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
        x, _ = self.encoder.predict(x)
        return {'x': x, 'y': y, 'p': p}
    def reconstruct(self, data):
        x, y, p = data['x'], data['y'], data['p']
        x, _ = self.encoder.predict(x)
        x = self.decoder.predict(x)
        return {'x': x, 'y': y, 'p': p}
    
@Privater.register('c_vae')
class CVae(Privater):
    def __init__(self,
                 img_dim=64,
                 z_dim=128,
                 p_dim=6,
                 rec_x_weight=64*64/10
                 ):
        def build_c_encoder():
            encoder = build_encoder(img_dim=img_dim,
                                    z_dim=2*z_dim,
                                    use_max_pooling=True, 
                                    drop_out=-1,
                                    use_gauss_prior=False)
            x_in = Input(shape=(img_dim, img_dim, 3))
            p_in = Input(shape=(p_dim,))
            z = encoder(x_in)
            z = Concatenate()([z, p_in])
            z_mean = Dense(z_dim)(z)
            z_log_var = Dense(z_dim)(z)
            return Model([x_in, p_in], [z_mean, z_log_var])
        encoder = build_c_encoder()
        x_in = Input(shape=(img_dim, img_dim, 3))
        p_in = Input(shape=(p_dim,))
        x = x_in
        z_mean, z_log_var = encoder([x, p_in])
        z = Lambda(gauss_sampling)([z_mean, z_log_var])
        gauss_loss = Lambda(gauss_loss_func, name='prior')([z_mean, z_log_var])
        decoder = build_decoder(img_dim=img_dim,
                                z_dim=z_dim+p_dim)
        rec_x = decoder(Concatenate()([z, p_in]))
        rec_x = Lambda(lambda x: x, name="rec_x")(rec_x)
        train_model = Model([x_in, p_in], [rec_x, gauss_loss])
        
        train_model.compile(optimizer=Adam(1e-3),
                            loss={'prior': identity_loss, 'rec_x': 'mean_squared_error'},
                            loss_weights={'prior':1, 'rec_x': rec_x_weight}
                            )
        self.encoder = encoder
        self.decoder = decoder
        self.train_model = train_model
    
    def get_input(self, data):
        x, y, p = data['x'], data['y'], data['p']
        return ([x, p], {'prior':x, 'rec_x':x})
    def predict(self, data):
        x, y, p = data['x'], data['y'], data['p']
        x, _ = self.encoder.predict([x, p])
        return {'x': x, 'y': y, 'p': p}
    def reconstruct(self, data):
        x, y, p = data['x'], data['y'], data['p']
        x, _ = self.encoder.predict([x, p])
        x = self.decoder.predict(np.concatenate([x, p], axis=-1))
        return {'x': x, 'y': y, 'p': p}
        
@Privater.register('gpf')
class Gpf(Privater):
    def __init__(self,
                 img_dim=64,
                 z_dim=128,
                 p_dim=6,
                 real_rec_x_weight=10,
                 fake_rec_z_weight=10,
                 judge_loss_weight=1
                 ):
        encoder = build_encoder(img_dim=img_dim,
                                z_dim=z_dim,
                                use_max_pooling=True, 
                                drop_out=-1,
                                use_gauss_prior=False)
        decoder = build_decoder(img_dim=img_dim,
                                z_dim=z_dim+p_dim)
        discriminator = build_discriminator(img_dim=img_dim,
                                            num_classes=p_dim,
                                            z_dim=z_dim)
        
        real_x_in = Input(shape=(img_dim, img_dim, 3))
        rnd_x_in = Input(shape=(img_dim, img_dim, 3))
        real_p_in = Input(shape=(p_dim,))
        fake_p_in = Input(shape=(p_dim,))
        
        expect_z = encoder(real_x_in)
        
        real_z = Concatenate()([expect_z, real_p_in])
        fake_z = Concatenate()([expect_z, fake_p_in])
        
        real_rec_x = decoder(real_z)
        real_rec_x = Lambda(lambda x: x, name="real_rec_x")(real_rec_x)# used for reconstruction error(maintain I(x;z))
        
        fake_rec_x = decoder(fake_z)
        fake_rec_z = encoder(fake_rec_x)
        def mean_squad_error(args):
            real, fake = args
            return K.mean(K.square(real - fake))
        fake_rec_z = Lambda(mean_squad_error, name='fake_rec_z')([expect_z, fake_rec_z])
        
        fake_judge, fake_classify = discriminator(fake_rec_x)
        rnd_judge, rnd_classify = discriminator(rnd_x_in)
        fake_classify = Lambda(lambda x: x, name='fake_classify')(fake_classify)
        rnd_classify = Lambda(lambda x: x, name='rnd_classify')(rnd_classify)
        
        #build g_train_model
        encoder.trainable = True
        decoder.trainable = True
        discriminator.trainable = False
        def g_judge_loss_func(args):
            real_score, fake_score = args
            return K.mean(real_score - fake_score)
        g_judge_loss = Lambda(g_judge_loss_func, name='g_judge_loss')([rnd_judge, fake_judge])
        g_train_model = Model([real_x_in, rnd_x_in, real_p_in, fake_p_in], 
                              [real_rec_x, fake_rec_z, fake_classify, rnd_classify, g_judge_loss])
        
        g_train_model.compile(optimizer=Adam(1e-3),
                            loss={'real_rec_x': 'mean_squared_error',
                                  'fake_rec_z': identity_loss,
                                  'fake_classify': 'categorical_crossentropy',
                                  'rnd_classify': 'categorical_crossentropy',
                                  'g_judge_loss': identity_loss},
                            loss_weights={'real_rec_x': real_rec_x_weight,
                                          'fake_rec_z': fake_rec_z_weight,
                                          'fake_classify': 1,
                                          'rnd_classify': 1,
                                          'g_judge_loss': judge_loss_weight},
                            metrics={'fake_classify': 'acc',
                                     'rnd_classify': 'acc'}
                            )
        
        #build d_train_model
        encoder.trainable = False
        decoder.trainable = False
        discriminator.trainable = True
        def d_judge_loss_func(args):
            real_score, fake_score, real_value, fake_value = args
            d_loss = real_score - fake_score
            d_loss = d_loss[:, 0]
            d_norm = 10 * K.mean(K.abs(real_value - fake_value), axis=[1, 2, 3])
            d_loss = K.mean(- d_loss + 0.5 * d_loss**2 / d_norm)
            return d_loss
        d_judge_loss = Lambda(d_judge_loss_func, name='d_judge_loss')([rnd_judge, fake_judge, rnd_x_in, fake_rec_x])
        d_train_model = Model([real_x_in, rnd_x_in, fake_p_in],
                              [fake_classify, rnd_classify, d_judge_loss]
                              )
        d_train_model.compile(optimizer=Adam(1e-3),
                              loss={'fake_classify': 'categorical_crossentropy',
                                    'rnd_classify': 'categorical_crossentropy',
                                    'd_judge_loss': identity_loss
                                   },
                              loss_weights={'fake_classify': 1,
                                           'rnd_classify': 1,
                                           'd_judge_loss': judge_loss_weight
                                          },
                              metrics={'fake_classify': 'acc',
                                       'rnd_classify': 'acc'}
                             )
        
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.g_train_model = g_train_model
        self.d_train_model = d_train_model
        self.train_model = g_train_model
    
    def gen_random_p(self, real_p):
        p = np.random.randint(real_p.shape[1], size=real_p.shape[0])
        return to_categorical(p, num_classes=real_p.shape[1])
    def get_input_d(self, data1, data2):
        x1, y1, p1 = data1['x'], data1['y'], data1['p']
        x2, y2, p2 = data2['x'], data2['y'], data2['p']
        fake_p = self.gen_random_p(p1)
        return ([x1, x2, fake_p], {'fake_classify':fake_p,
                                   'rnd_classify':p2,
                                   'd_judge_loss':x1
                                  })
    def get_input_g(self, data1, data2):
        x1, y1, p1 = data1['x'], data1['y'], data1['p']
        x2, y2, p2 = data2['x'], data2['y'], data2['p']
        fake_p = self.gen_random_p(p1)
        return ([x1, x2, p1, fake_p], {'real_rec_x': x1,
                                       'fake_rec_z': x1,
                                       'fake_classify': fake_p,
                                       'rnd_classify': p2,
                                       'g_judge_loss':x1
                                      })
        
    def predict(self, data):
        x, y, p = data['x'], data['y'], data['p']
        x = self.encoder.predict(x)
        return {'x': x, 'y': y, 'p': p}
    
    def shift(self, p, shift_step=1):
        ret = np.argmax(p, axis=-1).astype(np.int32)
        ret = (ret+shift_step)%(p.shape[1])
        ret = to_categorical(ret, num_classes=p.shape[1])
        #import pdb;pdb.set_trace()
        return ret
    def reconstruct(self, data):
        x, y, p = data['x'], data['y'], data['p']
        x = self.encoder.predict(x)
        fake_p = self.shift(p)
        #if x.ndim != p.ndim:
        #    import pdb;pdb.set_trace()
        x = self.decoder.predict(np.concatenate([x, fake_p], axis=-1))
        return {'x': x, 'y': y, 'p': p}
    
    
@Privater.register('gpf_zero_one')
class GpfZeroOne(Gpf):
    def gen_random_p(self, p):
        return self.shift(p)
    
    
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
