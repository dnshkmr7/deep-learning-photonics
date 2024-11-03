import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_losses, learning_rates):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label = 'Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label = 'Validation Loss')
    ax1.set_title(f'Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, learning_rates, 'g-', label = 'Learning Rate')
    ax2.set_title(f'Learning Rate Evolution')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('figures/training_history.png')
    plt.show()

def plot_comparison(predictions, targets, parameters, wavelengths):
    indices = np.random.choice(len(predictions), 6, replace = False)
    
    fig, axes = plt.subplots(2, 3, figsize = (15, 10))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        ax.plot(wavelengths[idx], targets[idx], 
                label = 'True Spectrum', 
                color = 'blue'
                )
        ax.plot(wavelengths[idx], predictions[idx], 
                label = 'Predicted Spectrum', 
                color = 'red', 
                linestyle = 'dashed'
                )
        
        params_text = f"λ0: {parameters[idx][0]:.1f}\nFWHM: {parameters[idx][1]:.1f}\nPp: {parameters[idx][2]:.1f}"
        ax.text(0.95, 0.95, params_text, 
                transform = ax.transAxes, 
                fontsize = 10, 
                horizontalalignment = 'right', 
                verticalalignment = 'top', 
                bbox = dict(facecolor = 'white', 
                            alpha = 0.5
                            )
                )
        
        ax.set_xlabel("Wavelengths (nm)")
        ax.set_ylabel("Spectral Intensity (dB)")
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc = 'upper center', 
               ncol = 2, 
               fontsize = 10
               )
    plt.tight_layout()
    
    plt.savefig('figures/spectra_comparison.png')
    plt.show()