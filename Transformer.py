from keras.models import load_model
from keras.layers import Layer, Dropout, LayerNormalization, Conv1D
from Attention import *

class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, num_heads, ff_dim, dropout = 0.1):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout)
        self.attn_normalize = LayerNormalization(input_shape = input_shape, epsilon = 1e-6)

        self.ff_conv1D_1 = Conv1D(filters = self.ff_dim, kernel_size = 1, activation = 'relu')
        self.ff_conv1D_2 = Conv1D(filters = input_shape[0][-1], kernel_size = 1) 
        self.ff_dropout = Dropout(self.dropout)
        self.ff_normalize = LayerNormalization(input_shape = input_shape, epsilon = 1e-6)
  
    def call(self, inputs):
        x = self.attn_multi(inputs)
        x = self.attn_dropout(x)
        x = self.attn_normalize(inputs[0] + x)

        x = self.ff_conv1D_1(x)
        x = self.ff_conv1D_2(x)
        x = self.ff_dropout(x)
        x = self.ff_normalize(inputs[0] + x)
        return x 

class TransformerDecoder(layers.Layer):
    def __init__(self, d_k, d_v, num_heads, ff_dim, dropout = 0.1):
        super(TransformerDecoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.attn_single =  Attention(self.d_k, self.d_v)
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout)
        self.attn_normalize = LayerNormalization(input_shape = input_shape, epsilon = 1e-6)

        
        self.attn_dropout = Dropout(self.dropout)
        self.attn_normalize = LayerNormalization(input_shape = input_shape, epsilon = 1e-6)

        self.ff_conv1D_1 = Conv1D(filters = self.ff_dim, kernel_size = 1, activation = 'relu')
        self.ff_conv1D_2 = Conv1D(filters = input_shape[0][-1], kernel_size=1) 
        self.ff_dropout = Dropout(self.dropout)
        self.ff_normalize = LayerNormalization(input_shape = input_shape, epsilon = 1e-6)

    def call(self, inputs):
        x = self.attn_single(inputs)
        x = self.attn_dropout(x)
        x = self.attn_normalize(inputs[0] + x)

        x = self.attn_multi(x)
        x = self.attn_dropout(x)
        x = self.attn_normalize(inputs[0] + x)

        x = self.ff_conv1D_1(x)
        x = self.ff_conv1D_2(x)
        x = self.ff_dropout(x)
        x = self.ff_normalize(inputs[0] + x)
        return x