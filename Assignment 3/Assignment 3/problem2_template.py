#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                     std=(0.5, 0.5, 0.5))])

device = "cuda:0" if torch.cuda.is_available() else "cpu"
kwargs = {} if device=='cpu' else {'num_workers': 1, 'pin_memory': True}

BATCH_SIZE = 4

MEAN = (0.5, 0.5, 0.5)
STD = (.25, .25, .25)

transform_train_aug = transforms.Compose([
	    transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(MEAN, STD)])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Adding some data augmentation
trainset_aug = torchvision.datasets.CIFAR10(root='./data',
                                  train=True,
                                  download=True,
                                  transform=transform_train_aug)

# Expanding our training set with our horizontaly flipped dataset
trainset = torch.utils.data.ConcatDataset((trainset, trainset_aug))

# Splitting the training set into a training and validation set
trainset, validationset = train_test_split(trainset, test_size=0.1) 

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, **kwargs)

validationloader = torch.utils.data.DataLoader(validationset, batch_size=BATCH_SIZE,
                                          shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
                                      
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, **kwargs)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    """ Structure of our CNN and DNN """
    def __init__(self):
        super(Net, self).__init__()

        # CNN MODEL:

        # CNN input layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32) # We added batch normalization
        self.dropout1 = nn.Dropout(p=0.1) # We added dropout
        
        # CNN layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.dropout2 = nn.Dropout(p=0.1)

        # CNN layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.dropout3 = nn.Dropout(p=0.1)

        # CNN layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(num_features=128)
        self.dropout4 = nn.Dropout(p=0.1)

        # CNN layer 5
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(num_features=32)

        # DNN MODEL:

        # DNN layer 1
        self.fc1 = nn.Linear(in_features=32*8*8, out_features=512)

        # DNN layer 2
        self.fc2 = nn.Linear(in_features=512, out_features=256)

        # DNN layer 3
        self.fc3 = nn.Linear(in_features=256, out_features=128)

        # DNN output layer
        self.fc4 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """Forward pass"""

        # Through CNN
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)
        # x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.batchnorm2(x)
        # x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        # x = self.pool3(x)
        x = self.batchnorm3(x)
        # x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.batchnorm4(x)
        # x = self.dropout4(x)

        x = F.relu(self.conv5(x))
        x = self.batchnorm5(x)

        # print(x.size())

        # Through DNN
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

if __name__ == '__main__':
    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    EPOCHS = 10

    for epoch in range(EPOCHS):
        # Loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch+1, running_loss/i))

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data in validationloader:
                images, labels = data[0].to(device), data[1].to(device)
                # images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print('Valdiation accuracy of the network on the validation images: %d %%' % (100 * val_correct / val_total))

    print('Finished Training')

    labels_total = []
    predicted_total = []

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            np_labels = labels.cpu().numpy()
            np_predicted = predicted.cpu().numpy()

            labels_total.extend(np_labels)
            predicted_total.extend(np_predicted)
        
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * test_correct / test_total))

    plt.figure(figsize=(9, 6))

    for i, c in enumerate(CLASSES):
        precision, recall, thresholds = precision_recall_curve(labels_total, predicted_total, pos_label=i)
        auprc = round(auc(recall, precision), ndigits=2)
        plt.plot(recall, precision, label=f'{c} with an AUPRC of {auprc}')
    
    plt.legend(loc='upper right')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    plt.grid()
    plt.show()
