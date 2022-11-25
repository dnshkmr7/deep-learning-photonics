import tensorflow as tf
from keras import layers
import numpy as np

class Attention(layers.Layer):
    def __init__(self, d_k, d_v):
        super(Attention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = layers.Dense(self.d_k, input_shape = input_shape, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')
        self.key = layers.Dense(self.d_k, input_shape = input_shape, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')
        self.value = layers.Dense(self.d_v, input_shape = input_shape, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')

    def call(self, inputs):
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b = True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis = -1)
    
        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out    
    
class MultiAttention(layers.Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(Attention(self.d_k, self.d_v))  
    
        self.linear = layers.Dense(input_shape[0][-1], input_shape = input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis = -1)
        multi_linear = self.linear(concat_attn)
        return multi_linear