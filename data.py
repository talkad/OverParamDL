import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
import torchvision
import numpy as np
import random
import torch



def create_dataset(corruption_percentage=0):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    num_samples = len(trainset)
    rands = np.random.choice(num_samples, int(num_samples * corruption_percentage), replace=False)

    for rand in rands:
        trainset.targets[rand] = np.random.choice(list(range(0, 10)))

    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.5 * num_train))

    # train_idx, eval_idx = indices[split:], indices[:split]
    train_idx, eval_idx = indices[split:], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    eval_sampler = SubsetRandomSampler(eval_idx)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=train_sampler, num_workers=4)
    trainloader_eval = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=eval_sampler, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    return trainloader, trainloader_eval, testloader