import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import numpy as np


EPOCHS = 50


def log_reader(log_lines, start=0, noise_rate=0, discrepancy=False):
    delta = 8

    logs = log_lines[start::delta]
    logs = [float(log[-5:]) for log in logs] if discrepancy else [1-float(log[-5:]) for log in logs]

    logs = np.array(logs).reshape(-1, EPOCHS)

    # remap_err = lambda v: 1.0 - (1-noise_rate)*(1-v) + v*noise_rate/9.0  # remap clean test error --> noisy test error.

    return logs.mean(axis=0), logs.std(axis=0)




if __name__=='__main__':

    plt.figure(figsize=(20, 10))
    plt.style.use('seaborn-v0_8-darkgrid')

    resnet_width = [8, 16, 32, 64]
    noises = [0.15, 0.4]  # [0, 0.2, 0.5]
    
    for idx, width in enumerate(resnet_width):
        
        plt.subplot(2, 2, idx+1)

        noise = 0.15
        log_file = f'logs9/resnet18_k={width}_noise={noise}.log'
        with open(log_file, 'r') as f:
            lines = f.readlines()

        test_discrepancy_mean, test_discrepancy_std= log_reader(lines, start=7, noise_rate=noise)

        epochs = range(1, EPOCHS+1)

        plt.plot(epochs, test_discrepancy_mean, label=f'noise={noise}', color='g')  
        plt.fill_between(epochs, test_discrepancy_mean-test_discrepancy_std, test_discrepancy_mean+test_discrepancy_std, color='g', alpha=0.3)

        noise = 0.4
        log_file = f'logs9/resnet18_k={width}_noise={noise}.log'
        with open(log_file, 'r') as f:
            lines = f.readlines()

        test_discrepancy_mean, test_discrepancy_std= log_reader(lines, start=7, noise_rate=noise)

        epochs = range(1, EPOCHS+1)

        plt.plot(epochs, test_discrepancy_mean, label=f'noise={noise}', color='r', linestyle='--')  
        plt.fill_between(epochs, test_discrepancy_mean-test_discrepancy_std, test_discrepancy_mean+test_discrepancy_std, color='r', alpha=0.3)

        plt.title(f'ResNet Width {width}')
        plt.xlabel('Epochs')
        plt.ylabel('Discrepancy')
        plt.legend()

    plt.savefig(f'ddd_augmentations.png')


