import torchvision
import random


def corrupt_labels(dataset, corruption_percentage=0):
    num_corrupt = int(corruption_percentage * len(dataset))
    corrupt_indices = random.sample(range(len(dataset)), num_corrupt)

    for idx in corrupt_indices:
        new_label = random.randint(0, 9)
        dataset.targets[idx] = new_label

    return dataset

def create_dataset(corruption_percentage=0):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset = corrupt_labels(trainset, corruption_percentage=corruption_percentage)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return trainset, testset
