import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from loader import *
from spectrumtransformer import *
from train import *

n_bins = 512
nheads = 16
nlayers = 6
initial_lr = 1e-4 
weight_decay = 1e-5
decay_factor = 0.5
scheduler_patience = 5  
batch_size = 16     
num_epochs = 100
model_patience = 10

param_dim = 3
input_dim = n_bins
output_dim = n_bins

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_supercontinuum = SpectrumTransformer(input_dim = input_dim, 
                                           param_dim = param_dim, 
                                           output_dim = output_dim, 
                                           nhead = nheads, 
                                           num_encoder_layers = nlayers)
model = model_supercontinuum

criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), 
                        lr = initial_lr, 
                        weight_decay = weight_decay
                        )
scheduler = ReduceLROnPlateau(optimizer, 
                              mode = 'min', 
                              factor = decay_factor, 
                              patience = scheduler_patience, 
                              verbose = True
                              )

train_data, val_data, test_data, scalers = load_data(n_bins)
train_loader = DataLoader(train_data, 
                          batch_size = batch_size, 
                          shuffle = True
                          )
val_loader = DataLoader(val_data, 
                        batch_size = batch_size, 
                        shuffle = False
                        )
test_loader = DataLoader(test_data, 
                         batch_size = batch_size, 
                         shuffle = False
                         )

train_losses, val_losses, learning_rates = train_model(
    model, 
    train_loader, val_loader, 
    criterion, optimizer, scheduler,
    num_epochs = num_epochs, 
    patience = model_patience
    )
avg_test_loss, _, _ = evaluate_model(model, test_loader, criterion, scalers)
print(f"Test Loss: {avg_test_loss:.4f}")