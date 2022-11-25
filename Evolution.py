import numpy as np
import time

def PredEvo(model, X_test, test_evo, steps, look_back, feat_len):
    # Make the time series
    evo_size = steps - 1
    Y_submit = np.zeros((test_evo, evo_size, feat_len))
    test_data = X_test[::evo_size,:,:]  # select fiber input profiles

    for step in range(evo_size):
        test_result = model.predict(test_data)
        Y_submit[:,step,:] = test_result
        test_result = np.expand_dims(test_result, axis=1)
        test_data = np.concatenate((test_data,test_result), axis=1)
        test_data = test_data[:, 1:, :]

    # reshape to the original dimensions
    Y_submit = np.reshape(Y_submit,(evo_size*test_evo, feat_len))

    return Y_submit


def PredEvoExpt(model, X_test, test_evo, steps, look_back, feat_len):
    # Make the time series
    evo_size = steps - look_back
    Y_submit = np.zeros((test_evo, evo_size, feat_len))
    test_data = X_test[::evo_size, :, :]  # select fiber input profiles

    for step in range(evo_size):
        test_result = model.predict(test_data)
        Y_submit[:, step, :] = test_result
        test_result = np.expand_dims(test_result, axis=1)
        test_data = np.concatenate((test_data,test_result), axis=1)
        test_data = test_data[:, 1:, :]

    # reshape to the original dimensions
    Y_submit = np.reshape(Y_submit,(evo_size*test_evo, feat_len))

    return Y_submit

def PredEvoAddP(model, X_test, test_evo, steps, look_back, added_params, feat_len):
    # Make the time series
    evo_size = steps - 1
    Y_submit = np.zeros((test_evo, evo_size, feat_len))
    test_data = X_test[::evo_size, :, :]  # select fiber input profiles

    for step in range(evo_size):
        test_result = model.predict(test_data)
        Y_submit[:, step, :] = test_result
        ap = test_data[:, 0, :added_params]
        # pass the additional variables for the prediction
        test_result = np.concatenate((ap, test_result), axis=1)
        test_result = np.expand_dims(test_result, axis=1)
        test_data = np.concatenate((test_data,test_result), axis=1)
        test_data = test_data[:, 1:, :]

    # reshape to the original dimensions
    Y_submit = np.reshape(Y_submit,(evo_size*test_evo, feat_len))

    return Y_submit
