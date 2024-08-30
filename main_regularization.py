import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import random
from data import create_dataset
import logging
from tqdm import tqdm
import itertools
from resnet import make_resnet18k
from torchvision.models import resnet18





def ddd(trainloader, trainloader_eval, testloader, net1, net2, criterion1, criterion2, optimizer1, optimizer2, logger, epochs=100, b=0.1):
    for epoch in range(epochs):
        total_loss1, total_loss2, total = 0, 0, 0
        
        # TRAIN
        net1.train()
        net2.train()

        for inputs, labels in tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer1.zero_grad()
            outputs1 = net1(inputs)
            loss1 = (criterion1(outputs1, labels) - b).abs() + b 
            loss1.backward()
            optimizer1.step()
            total_loss1 += loss1.item()

            optimizer2.zero_grad()
            outputs2 = net2(inputs)
            loss2 = (criterion2(outputs2, labels) - b).abs() + b 
            loss2.backward()
            optimizer2.step()
            total_loss2 += loss2.item()

            total += 1
        
        logger.info(f'Epoch {epoch} | Net1 Total Loss: {total_loss1 / total:.3f}')
        logger.info(f'Epoch {epoch} | Net2 Total Loss: {total_loss2 / total:.3f}')

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

                    total_discrepancy += (preds1 == preds2).sum().item()

            accuracy1 = correct1 / total
            accuracy2 = correct2 / total
            average_discrepancy = total_discrepancy / total

            logger.info(f'Epoch {epoch} | Net1 {split} Accuracy: {accuracy1:.3f}')
            logger.info(f'Epoch {epoch} | Net2 {split} Accuracy: {accuracy2:.3f}')
            logger.info(f'Epoch {epoch} | Discrepancy: {average_discrepancy:.3f}')

    return net1, net2




if __name__=='__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    repeats = 5
    noise_ratios = [0, 0.15, 0.4]
    resnet_widths = [1] + [num for num in range(2, 23) if num % 2 == 0] + [num for num in range(24, 65) if num % 4 == 0]

    bs = [0, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 2]
    resnet_width = 64

    for idx in range(repeats):
        for b, noise_ratio  in itertools.product(bs, noise_ratios):
        
            # for noise_ratio  in noise_ratios:

            print(f'Noise Ratio: {noise_ratio}, b: {b}')

            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            if logger.hasHandlers():
                logger.handlers.clear()

            file_handler = logging.FileHandler(f'logs10/resnet18_b={b}_noise={noise_ratio}.log')
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            trainloader, trainloader_eval, testloader = create_dataset(corruption_percentage=noise_ratio)

            torch.manual_seed(random.randint(0, 999999))
            net1 = make_resnet18k(k=resnet_width, num_classes=10)
            net1 = net1.to(device)

            torch.manual_seed(random.randint(0, 999999))
            net2 = make_resnet18k(k=resnet_width, num_classes=10)
            net2 = net2.to(device)

            criterion1 = nn.CrossEntropyLoss()
            optimizer1 = optim.SGD(net1.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) # optim.SGD(net1.parameters(), lr=0.1, momentum=0.9) 

            criterion2 = nn.CrossEntropyLoss()
            optimizer2 = optim.SGD(net2.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) # optim.SGD(net2.parameters(), lr=0.1, momentum=0.9) 

            ddd(trainloader, trainloader_eval, testloader, net1, net2, criterion1, criterion2, optimizer1, optimizer2, logger, epochs=50, b=b)
