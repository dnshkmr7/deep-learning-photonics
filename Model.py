from datetime import datetime
from time import time

import tensorflow as tf
from keras.layers import GlobalAveragePooling1D, Dense, Dropout
from keras import Input
from keras.models import Model
from Transformer import *
from Time2Vec import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score

class ModelBlock(object):
    def __init__(self, d_k, d_v, num_heads, ff_dim, look_back, feat_len, added_params):
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.look_back = look_back
        self.feat_len = feat_len
        self.added_params = added_params

    def BuildModel(self):
        time_embedding = Time2Vector(self.look_back)
        encoder_layer = TransformerEncoder(self.d_k, self.d_v, self.num_heads, self.ff_dim)
        decoder_layer = TransformerDecoder(self.d_k, self.d_v, self.num_heads, self.ff_dim)

        inputs = Input(shape = (self.look_back, self.feat_len + self.added_params))
        x = time_embedding(inputs)
        x = tf.concat([inputs, x], -1)
        x = encoder_layer((x,x,x))
        x = decoder_layer((x,x,x))
        x = GlobalAveragePooling1D(data_format = "channels_first")(x)
        x = Dropout(0.1)(x)
        x = Dense(250, activation = 'relu')(x)
        x = Dropout(0.1)(x)
        outputs = layers.Dense(self.feat_len, activation = 'sigmoid')(x)
       
        model = Model(inputs = inputs, outputs = outputs)
        return model

    def Train(self, X_train, y_train, epochs = 100, batch_size = 64):
        self.model = self.BuildModel()
        self.model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse', 'mae'])
        print(self.model.summary())

        # Stop training if error does not improve within 50 iterations
        early_stopping_monitor = EarlyStopping(patience = 50, restore_best_weights = True)

        # Save the best model ... with minimal error
        filepath = "./TNN_pred_best"+datetime.now().strftime('%d%m%Y_%H:%M:%S')+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 2, save_best_only = True, mode = 'min')

        callback_history = self.model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.2, verbose = 2,
                                callbacks=[early_stopping_monitor, checkpoint])
                                #callbacks=[PlotLossesKeras(), early_stopping_monitor, checkpoint])

    def Evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        _, rmse_result, mae_result, smape_result, _ = self.model.evaluate(X_test, y_test)
        r2_result = r2_score(y_test.flatten(), y_pred.flatten())
        return _, rmse_result, mae_result, smape_result, r2_result
