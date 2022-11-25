import numpy as np
from matplotlib import pyplot as plt

def plot_history(history, timestr, add_time):
    plt.figure()
    plt.xlabel('Epoch')
    plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
    plt.legend()
    #plt.show()
    if add_time:
        fname = 'nets/training_'+timestr+'.png'
    else:
        fname = 'nets/training.png'
    plt.savefig(fname)