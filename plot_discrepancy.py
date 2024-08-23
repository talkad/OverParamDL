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

    resnet_width = 64
    noises = [0, 0.15, 0.4]  # [0, 0.2, 0.5]
    resnet_vers = [18] #[18, 34, 50, 101, 152]
    
    for resnet_ver, noise in itertools.product(resnet_vers, noises):
        plt.clf()

        log_file = f'logs7/resnet{resnet_ver}_k={resnet_width}_noise={noise}.log'

        with open(log_file, 'r') as f:
            lines = f.readlines()

        train_accuracy1, _ = log_reader(lines, start=2, noise_rate=noise)
        train_accuracy2, _ = log_reader(lines, start=3, noise_rate=noise)

        test_accuracy1, _  = log_reader(lines, start=5, noise_rate=noise)
        test_accuracy2, _  = log_reader(lines, start=6, noise_rate=noise)

        test_discrepancy_mean, test_discrepancy_std= log_reader(lines, start=7, noise_rate=noise)

        epochs = range(1, EPOCHS+1)

        plt.figure(figsize=(20, 5))
        plt.style.use('seaborn-v0_8-darkgrid')


        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_accuracy1, label='train net1')   
        plt.plot(epochs, test_accuracy1, label='test net1')   

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_accuracy2, label='train net2')   
        plt.plot(epochs, test_accuracy2, label='test net2')   

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, test_discrepancy_mean, label=f'noise={noise}')  
        plt.fill_between(epochs, test_discrepancy_mean-test_discrepancy_std, test_discrepancy_mean+test_discrepancy_std, alpha=0.3)

        plt.xlabel('Epochs')
        plt.ylabel('Discrepancy')
        plt.legend()

        plt.savefig(f'ddd_k={resnet_width}_noise={noise}.png')


