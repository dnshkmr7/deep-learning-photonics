import numpy as np
import scipy.io as sio

def LoadData(filename, train_evo, test_evo, steps, look_back, normalization = 'none'):
    mat_contents = sio.loadmat(filename)
    data = mat_contents['data']
    print("data loaded...")
    print(data.shape)

    if normalization == 'none':
        pass
        
    elif normalization == 'max':
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data = data/m_max

    elif normalization == 'dBm':
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data=data/m_max
        data = 10*np.log10(data)
        dBlim = 55
        data[data < -dBlim] = -dBlim
        data = data/dBlim + 1

    elif normalization == 'manual':
        m_max = 10369993.175721595
        print('max:', m_max)
        data = data/m_max

    feat_len = data.shape[1]

    num_evo = train_evo + test_evo
    evo_size = steps - 1
    num_samples = np.round(num_evo*evo_size).astype(int)
    X_data_series = np.zeros((num_samples, look_back, feat_len))
    Y_data_series = np.zeros((num_samples, feat_len))

    for evo in range(num_evo):
        evo_data = np.transpose(data[evo, :, :])
        temp1 = evo_data[0, :]
        temp2 = np.tile(temp1, (look_back - 1, 1))
        evo_data = np.vstack((temp2, evo_data))

        for step in range(evo_size):
            input_data = evo_data[step:step + look_back, :]
            output_data = evo_data[step + look_back, :]
            series_idx = evo*evo_size + step
            X_data_series[series_idx, :, :] = input_data
            Y_data_series[series_idx, :] = output_data


    X_train = X_data_series[:num_samples - test_evo*evo_size]
    X_test = X_data_series[num_samples - test_evo*evo_size:]
    Y_train = Y_data_series[:num_samples - test_evo*evo_size]
    Y_test = Y_data_series[num_samples - test_evo*evo_size:]

    return feat_len, X_train, X_test, Y_train, Y_test


def LoadDataExpt(filename, train_evo, test_evo, steps, look_back, normalization = 'none'):
    mat_contents = sio.loadmat(filename)
    data = mat_contents['data']
    print("data loaded...")
    print(data.shape)

    if normalization == 'none':
        pass

    elif normalization == 'max':
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data = data/m_max

    elif normalization == 'dBm':
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data=data/m_max
        data = 10*np.log10(data)
        dBlim = 55
        data[data < -dBlim] = -dBlim
        data = data/dBlim + 1

    elif normalization == 'manual':
        m_max = 10369993.175721595
        print('max:', m_max)
        data = data/m_max

    feat_len = data.shape[1]

    num_evo = train_evo + test_evo
    evo_size = steps - look_back
    num_samples = np.round(num_evo*evo_size).astype(int)
    X_data_series = np.zeros((num_samples, look_back, feat_len))
    Y_data_series = np.zeros((num_samples, feat_len))

    for evo in range(num_evo):
        for step in range(evo_size):
            input_data = np.transpose(data[evo, :, step:step + look_back])
            output_data = data[evo, :, step + look_back]
            series_idx = evo*evo_size + step
            X_data_series[series_idx,:,:] = input_data
            Y_data_series[series_idx,:] = output_data

    X_train = X_data_series[:num_samples - test_evo*evo_size]
    X_test = X_data_series[num_samples - test_evo*evo_size:]
    Y_train = Y_data_series[:num_samples - test_evo*evo_size]
    Y_test = Y_data_series[num_samples - test_evo*evo_size:]

    return feat_len, X_train, X_test, Y_train, Y_test

def LoadDataAddP(filename, train_evo, test_evo, steps, look_back, added_params, normalization='none'):
    mat_contents = sio.loadmat(filename)
    data = mat_contents['data']
    print("data loaded...")
    print(data.shape)

    if normalization == 'none':
        pass

    elif normalization == 'max':
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data = data/m_max

    elif normalization == 'maxC':
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data[:, added_params:, :] = data[:, added_params:, :]/m_max
        data[:, :added_params, :] = (data[:, :added_params, :] + 8)/16

    elif normalization == 'dBmQ':
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data[:, added_params:, :] = data[:, added_params:, :]/m_max
        data[:, added_params:, :] = 10*np.log10(data[:, added_params:, :])
        dBlim = 55
        data[data < -dBlim] = -dBlim
        data[:, added_params:, :] = data[:, added_params:, :]/dBlim + 1
        data[:, :added_params, :] = data[:, :added_params, :]/9

    elif normalization == 'dBmCC':
        m_max = np.max(np.fabs(data))
        print('max:', m_max)
        data[:, added_params:, :] = data[:, added_params:, :]/m_max
        data[:, added_params:, :] = 10*np.log10(data[:, added_params:, :])
        data[data < -dBlim] = -dBlim
        data[:, added_params:, :] = data[:, added_params:, :]/dBlim + 1

    elif normalization == 'manual':
        m_max = 10369993.175721595
        print('max:', m_max)
        data = data/m_max

    feat_len = data.shape[1] - added_params

    num_evo = train_evo + test_evo
    evo_size = steps - 1
    num_samples = np.round(num_evo*evo_size).astype(int)
    X_data_series = np.zeros((num_samples, look_back, feat_len + added_params))
    Y_data_series = np.zeros((num_samples, feat_len))

    for evo in range(num_evo):
        evo_data = np.transpose(data[evo, :, :])
        temp1 = evo_data[0, :]
        temp2 = np.tile(temp1, (look_back - 1,1))
        evo_data = np.vstack((temp2, evo_data))

        for step in range(evo_size):
            input_data = evo_data[step:step + look_back, :]
            output_data = evo_data[step + look_back, added_params:]
            series_idx = evo*evo_size + step
            X_data_series[series_idx, :, :] = input_data
            Y_data_series[series_idx, :] = output_data

    X_train = X_data_series[:num_samples - test_evo*evo_size]
    X_test = X_data_series[num_samples - test_evo*evo_size:]
    Y_train = Y_data_series[:num_samples - test_evo*evo_size]
    Y_test = Y_data_series[num_samples - test_evo*evo_size:]

    return feat_len, X_train, X_test, Y_train, Y_test
