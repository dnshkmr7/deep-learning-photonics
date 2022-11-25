import tensorflow as tf
from keras.layers import Layer

class Time2Vector(Layer):
    def __init__(self, look_back, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = look_back

    def build(self, input_shape):
        self.w_l = self.add_weight(name = 'wl', shape = (int(self.seq_len),), initializer = 'uniform', trainable = True)
        self.b_l = self.add_weight(name = 'bl', shape = (int(self.seq_len),), initializer = 'uniform', trainable = True)
    
        self.w_p = self.add_weight(name = 'wp', shape = (int(self.seq_len),), initializer = 'uniform', trainable = True)
        self.b_p = self.add_weight(name = 'bp', shape = (int(self.seq_len),), initializer = 'uniform', trainable = True)

    def call(self, x):
        x = tf.math.reduce_mean(x[:,:,:4], axis = -1) 
        t_l = self.w_l * x + self.b_l
        t_l = tf.expand_dims(t_l, axis=-1)
    
        t_p = tf.math.sin(tf.multiply(x, self.w_p) + self.b_p)
        t_p = tf.expand_dims(t_p, axis = -1)
        return tf.concat([t_l, t_p], axis = -1)