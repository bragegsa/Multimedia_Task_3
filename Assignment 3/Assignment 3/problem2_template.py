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

batch_size = 4

mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)

transform_train_aug = transforms.Compose([
	    transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# We added som data augmentation
trainset_aug = torchvision.datasets.CIFAR10(root='./data',
                                  train=True,
                                  download=True,
                                  transform=transform_train_aug)

# Here we expand our training set with our horizontaly flipped dataset
trainset = torch.utils.data.ConcatDataset((trainset, trainset_aug))

trainset, validationset = train_test_split(trainset, test_size=0.1) # Splitting the training set into a training and validation set

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, **kwargs)

validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                          shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
                                       
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, **kwargs)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32) # We added batch normalization
        self.dropout1 = nn.Dropout(p=0.1) # We added dropout
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.dropout3 = nn.Dropout(p=0.1)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(num_features=32)

        # DNN
        self.fc1 = nn.Linear(in_features=32*8*8, out_features=120)

        self.fc2 = nn.Linear(in_features=120, out_features=84)

        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):

        # CNN:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)
        # x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        # x = self.pool3(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.batchnorm4(x)

        # print(x.size())

        # DNN:
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = 10

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / i))

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

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            np_labels = labels.cpu().numpy()
            np_predicted = predicted.cpu().numpy()

            labels_total.extend(np_labels)
            predicted_total.extend(np_predicted)
        
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    plt.figure(figsize=(9, 6))

    for i in range(len(classes)):
        precision, recall, thresholds = precision_recall_curve(labels_total, predicted_total, pos_label=i)
        auprc = round(auc(recall, precision), ndigits=2)
        plt.plot(recall, precision, label=f'{classes[i]} with an AUPRC of {auprc}')
    
    plt.legend(loc='upper right')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    plt.grid()
    plt.show()
