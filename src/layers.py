from keras.layers import Dense, Layer

class VIB(Layer):
    """变分信息瓶颈层
    """
    def __init__(self, lamb, **kwargs):
        self.lamb = lamb
        super(VIB, self).__init__(**kwargs)
    def call(self, inputs):
        z_mean, z_log_var = inputs
        u = K.random_normal(shape=K.shape(z_mean))
        kl_loss = - 0.5 * K.sum(K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), 0))
        self.add_loss(self.lamb * kl_loss)
        u = K.in_train_phase(u, 0.)
        return z_mean + K.exp(z_log_var / 2) * u
    def compute_output_shape(self, input_shape):
        return input_shape[0]
