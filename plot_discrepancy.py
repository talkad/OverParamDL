import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import numpy as np


def log_reader(log_lines, start=0, discrepancy=False):
    delta = 6

    logs = log_lines[start::delta]
    logs = [float(log[-5:]) for log in logs] if discrepancy else [1-float(log[-5:]) for log in logs]

    logs = np.array(logs).reshape(-1, 50)

    # return logs.mean(axis=0), logs.min(axis=0), logs.max(axis=0)
    return logs.mean(axis=0), logs.apply(lambda x: x.nlargest(2).iloc[-1], axis=0)
                            , logs.apply(lambda x: x.nlargest(2).iloc[-1], axis=0)




if __name__=='__main__':

    resnet_width = 64
    noises = [0, 0.2, 0.5]
    resnet_vers = [18] #[18, 34, 50, 101, 152]
    
    for resnet_ver, noise in itertools.product(resnet_vers, noises):
        plt.clf()

        log_file = f'logs_original_subset02/resnet{resnet_ver}_noise={noise}_k={resnet_width}.log'

        with open(log_file, 'r') as f:
            lines = f.readlines()

        train_accuracy1, _, _ = log_reader(lines, start=0)
        train_accuracy2, _, _ = log_reader(lines, start=1)

        test_accuracy1, _, _  = log_reader(lines, start=3)
        test_accuracy2, _, _  = log_reader(lines, start=4)

        test_discrepancy, mind, maxd = log_reader(lines, start=2, discrepancy=True)

        epochs = range(1, 51)

        plt.figure(figsize=(20, 5))
        plt.style.use('seaborn-v0_8-darkgrid')


        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_accuracy1, label='train net1')   
        plt.plot(epochs, test_accuracy1, label='test net1')   

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        # plt.grid()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_accuracy2, label='train net2')   
        plt.plot(epochs, test_accuracy2, label='test net2')   

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        # plt.grid()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, test_discrepancy, label=f'noise={noise}')  
        # y_err = np.array([test_discrepancy - mind, maxd - test_discrepancy]) 
        plt.fill_between(epochs, mind, maxd, alpha=0.3)

        plt.xlabel('Epochs')
        plt.ylabel('Discrepancy')
        plt.legend()
        # plt.grid()

        # plt.savefig(f'ddd/resnet{resnet_ver}/subset0.2_ddd_noise={noise}.png')
        plt.savefig(f'subset0.2_ddd_noise={noise}.png')


