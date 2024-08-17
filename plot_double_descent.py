import matplotlib.pyplot as plt
import numpy as np


def log_reader(log_lines, start=0):
    delta = 6

    logs = log_lines[start::delta]
    return [float(log[-5:]) for log in logs]


if __name__=='__main__':

    resnet_widths = [1] + [num for num in range(2, 23) if num % 2 == 0] + [num for num in range(24, 65) if num % 4 == 0]

    epochs = range(18, 23, 2)
    epochs = range(45, 49, 2)


    plt.figure(figsize=(15, 5))
    plt.style.use('seaborn-v0_8-darkgrid')

    for noise in [0, 0.2, 0.5]:   
        dd_total = []

        for epoch in epochs:
            dd_test = []

            for k in resnet_widths:
                with open(f'logs2/resnet18_noise={noise}_k={k}.log', 'r') as f:
                    lines = f.readlines()

                    test_accuracy1 = log_reader(lines, start=3)
                    # test_accuracy1 = log_reader(lines, start=0)
                    dd_test.append(1 - test_accuracy1[epoch])
            
            dd_total.append(dd_test)

        dd_total = np.array(dd_total)
        plt.plot(resnet_widths[1:], dd_total.mean(axis=0)[1:], label=f'Noise {noise}')   
        plt.fill_between(resnet_widths[1:], dd_total.min(axis=0)[1:], dd_total.max(axis=0)[1:], alpha=0.3)

    plt.xlabel('ResNet Width (K)') 
    plt.ylabel('Error')   
    plt.legend()

    plt.savefig(f'dd.png')
    # plt.savefig(f'dd_epoch={epoch}_train.png')


