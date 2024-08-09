import matplotlib.pyplot as plt


def log_reader(log_lines, start=0):
    delta = 6

    logs = log_lines[start::delta]
    return [float(log[-5:]) for log in logs]


if __name__=='__main__':

    log_file = 'resnet18_noise=0.2_k=64_3.log'

    with open(log_file, 'r') as f:
        lines = f.readlines()

    train_accuracy1 = log_reader(lines, start=0)
    train_accuracy2 = log_reader(lines, start=1)

    test_accuracy1 = log_reader(lines, start=3)
    test_accuracy2 = log_reader(lines, start=4)

    test_discrepancy = log_reader(lines, start=5)

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_accuracy1, label='train net1')   
    plt.plot(test_accuracy1, label='test net1')   

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracy2, label='train net2')   
    plt.plot(test_accuracy2, label='test net2')   

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(test_discrepancy, label='noise 20%')   

    plt.xlabel('Epochs')
    plt.ylabel('Discrepancy')
    plt.legend()
    plt.grid()

    plt.savefig('resnet18_k=64_3.png')

