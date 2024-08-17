import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# from resnet import ResNet18
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import numpy as np
import random
from data import create_dataset
import logging
from tqdm import tqdm
import itertools



def ddd(trainloader, trainloader_eval, testloader, net1, net2, criterion, optimizer1, optimizer2, scheduler1, scheduler2, logger, epochs=50):
    for epoch in range(epochs):
        
        # TRAIN
        net1.train()
        net2.train()

        for inputs, labels in tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer1.zero_grad()
            outputs1 = net1(inputs)
            loss1 = criterion(outputs1, labels)
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            outputs2 = net2(inputs)
            loss2 = criterion(outputs2, labels)
            loss2.backward()
            optimizer2.step()

        scheduler1.step()
        scheduler2.step()

        # EVAL
        net1.eval()
        net2.eval()
        
        for split, dataloader in [('Train', trainloader_eval), ('Test', testloader)]:
            correct1, correct2, total = 0, 0, 0
            total_discrepancy = 0

            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    total += labels.size(0)

                    outputs1 = net1(inputs)
                    outputs2 = net2(inputs)

                    preds1 = torch.argmax(outputs1, dim=1)
                    preds2 = torch.argmax(outputs2, dim=1)

                    correct1 += preds1.eq(labels).sum().item()
                    correct2 += preds2.eq(labels).sum().item()

                    discrepancy = (preds1 != preds2).sum().item()
                    total_discrepancy += discrepancy

            accuracy1 = correct1 / total
            accuracy2 = correct2 / total
            average_discrepancy = total_discrepancy / total

            logger.info(f'Epoch {epoch} | Net1 {split} Accuracy: {accuracy1:.3f}')
            logger.info(f'Epoch {epoch} | Net2 {split} Accuracy: {accuracy2:.3f}')
            logger.info(f'Epoch {epoch} | Discrepancy: {average_discrepancy:.3f}')

    return net1, net2




if __name__=='__main__':

    noise_ratios = [0, 0.2, 0.5]
    # resnet_widths = [1] + [num for num in range(2, 23) if num % 2 == 0] + [num for num in range(24, 65) if num % 4 == 0]

    # for resnet_width, noise_ratio  in itertools.product(resnet_widths, noise_ratios):
    resnet_width = 64
    for idx in range(10):
        for noise_ratio  in noise_ratios:

            print(f'Noise Ratio: {noise_ratio}, ResNet Width: {resnet_width}')

            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            if logger.hasHandlers():
                logger.handlers.clear()

            file_handler = logging.FileHandler(f'logs_original_subset02/resnet18_noise={noise_ratio}_k={resnet_width}.log')
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            trainset, testset = create_dataset(corruption_percentage=noise_ratio)

            split_size = int(len(trainset) * 0.2)
            trainset, _ = torch.utils.data.random_split(trainset, [split_size, len(trainset) - split_size])


            # split_size = int(len(trainset) * 0.5)
            # train_subset, _ = torch.utils.data.random_split(trainset, [split_size, len(trainset) - split_size])

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
            trainloader_eval = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=4)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

            torch.manual_seed(0)
            net1 = resnet18(num_classes=10).cuda() # ResNet18(width=resnet_width).cuda()

            torch.manual_seed(1)
            net2 = resnet18(num_classes=10).cuda() # ResNet18(width=resnet_width).cuda()

            criterion = nn.CrossEntropyLoss()
            optimizer1 = optim.SGD(net1.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.1)

            optimizer2 = optim.SGD(net2.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.1)

            net1, net2 = ddd(trainloader, trainloader_eval, testloader, net1, net2, criterion, optimizer1, optimizer2, scheduler1, scheduler2, logger)

            # torch.save(net1.state_dict(), 'bin/resnet18_1.pt')
            # torch.save(net2.state_dict(), 'bin/resnet18_2.pt')
