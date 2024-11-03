import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SpectrumDataset(Dataset):
    def __init__(self, parameters, wavelengths, spectra):
        self.parameters = torch.tensor(parameters, 
                                       dtype = torch.float32
                                       )
        self.wavelengths = torch.tensor(wavelengths, 
                                        dtype = torch.float32
                                        )
        self.spectra = torch.tensor(spectra, 
                                    dtype = torch.float32
                                    )
    
    def __len__(self):
        return len(self.parameters)
    
    def __getitem__(self, idx):
        return self.parameters[idx], self.wavelengths[idx], self.spectra[idx]

def bin_spectra_waves(spectra, wavelengths, n_bins):
    bin_size = spectra.shape[1] // n_bins
    
    reshaped_spectra = spectra.reshape(spectra.shape[0], n_bins, bin_size)
    reshaped_wavelengths = wavelengths.reshape(wavelengths.shape[0], n_bins, bin_size)
    
    integrated_spectra = np.trapz(reshaped_spectra, 
                                  reshaped_wavelengths, 
                                  axis = 2
                                  )
    avg_spectra = integrated_spectra / (reshaped_wavelengths[:, :, -1] - reshaped_wavelengths[:, :, 0])
    
    avg_wavelengths = reshaped_wavelengths.mean(axis = 2)
    
    return avg_spectra, avg_wavelengths

def load_data(n_bins = 512):
    parameters = pd.read_csv(f'data/parameters.csv').values
    wavelengths = pd.read_csv(f'data/wavelength.csv').values
    spectra = pd.read_csv(f'data/spectra.csv').values

    spectra, wavelengths = bin_spectra_waves(spectra, wavelengths, n_bins)

    params_scaler, wave_scaler, spectra_scaler = StandardScaler(), StandardScaler(), StandardScaler()

    train_params, test_params, train_waves, test_waves, train_spectra, test_spectra = train_test_split(
        parameters, 
        wavelengths, 
        spectra, 
        test_size = 0.1, 
        random_state = 11011
        )
    
    train_params, val_params, train_waves, val_waves, train_spectra, val_spectra = train_test_split(
        train_params, 
        train_waves, 
        train_spectra, 
        test_size = 0.2, 
        random_state = 11011
        )

    train_params = params_scaler.fit_transform(train_params)
    val_params = params_scaler.transform(val_params)
    test_params = params_scaler.transform(test_params)
    
    train_waves = wave_scaler.fit_transform(train_waves)
    val_waves = wave_scaler.transform(val_waves)
    test_waves = wave_scaler.transform(test_waves)
    
    train_spectra = spectra_scaler.fit_transform(train_spectra)
    val_spectra = spectra_scaler.transform(val_spectra)
    test_spectra = spectra_scaler.transform(test_spectra)

    scalers = {'parameters': params_scaler, 
               'wavelengths': wave_scaler, 
               'spectra': spectra_scaler
               }

    train_data = SpectrumDataset(train_params, 
                                 train_waves, 
                                 train_spectra
                                 )
    val_data = SpectrumDataset(val_params, 
                               val_waves, 
                               val_spectra
                               )
    test_data = SpectrumDataset(test_params, 
                                test_waves, 
                                test_spectra
                                )
    
    return train_data, val_data, test_data, scalers