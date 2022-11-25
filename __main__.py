import sys
import time
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.optimizers import Adam

from Load import *
from Model import *
from Evolution import *
from Utils import plot_history

if __name__ == '__main__':

	np.random.seed(123) 
	print(sys.version)
	start = time.time()
	add_time = 0

	num_epoch = 50 # number of epochs
	look_back = 10 # Lookback window size
	d_k = 256
	d_v = 256
	n_heads = 4
	ff_dim = 4


	# select data file
	#filename = 'simulations/HOS_NLSE_time_145.mat' # train_evo=2900, test_evo=100, steps=101, i_x=145
	#filename = 'simulations/HOS_NLSE_spec_126.mat' # train_evo=2900, test_evo=100, steps=101, i_x=145
	#filename = 'simulations/HOS_expt_time_151.mat' # train_evo=2899, test_evo=100, steps=110, i_x=151
	#filename = 'simulations/HOS_expt_spec_126.mat' # train_evo=2899, test_evo=100, steps=110, i_x=126
	#filename = 'simulations/SC_time_276.mat' # train_evo=1250, test_evo=50, steps=200, i_x=276
	filename = '/home/disno/RNNnonlinear_v2/simulations/SC_spec_251.mat' # train_evo=1250, test_evo=50, steps=200, i_x=251
	#filename = 'simulations/norm_NLSE_time_256.mat' # train_evo=950, test_evo=50, steps=101, i_x=256
	#filename = 'simulations/norm_NLSE_spec_128.mat' # train_evo=950, test_evo=50, steps=101, i_x=128
	#filename = 'simulations/chirped_NLSE_time_256.mat' # train_evo=5900, test_evo=100, steps=101, i_x=256, added_params=10
	#filename = 'simulations/norm_GNLSE_spec_132.mat' # train_evo=11800, test_evo=200, steps=51, i_x=132, added_params=10
	#filename = 'simulations/MMGNLSE_spec_301.mat' # train_evo=950, test_evo=50, steps=100, i_x=256, added_params=25


	# define samples for training and testing, and the number of propagation
	# steps in the evolution
	train_evo, test_evo, steps = 1250, 50, 200
	batch_size = test_evo + train_evo

	# define the number of added parameters for chirped_NLSE_time_256 (10),
	# norm_GNLSE_spec_132 (10) and MMGNLSE_spec_301 (25). 0 for other cases.
	added_params = 0

	# load data
	feat_len, X_train, X_test, Y_train, Y_test = LoadData(filename, train_evo, test_evo, steps, look_back,'dBm') # max/dBm

	# load_data_expt is used with HOS_expt_time_151 and HOS_expt_spec_126
	#feat_len, X_train, X_test, Y_train, Y_test = LoadDataExpt(filename, train_evo, test_evo, steps, look_back,'max') # max/dBm

	# load_data_addP is used with chirped_NLSE_time_256, norm_GNLSE_spec_132 and MMGNLSE_spec_301
	#feat_len, X_train, X_test, Y_train, Y_test = LoadDataAddP(filename, train_evo, test_evo, steps, look_back, added_params, 'maxC') # maxC/dBmQ/dBmCC

	print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
	print("READY...")
	block = ModelBlock(d_k = d_k, d_v = d_v, num_heads = 4, ff_dim = 4, look_back = look_back, feat_len = feat_len, added_params = added_params)
	block.Train(X_train, Y_train, batch_size)
	_, rmse_result, mae_result, smape_result, r2_result = block.Evaluate(X_test, Y_test)