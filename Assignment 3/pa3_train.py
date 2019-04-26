#Justin Powell
#PA3
#Robot Vision - Spring 2019

import os
import os.path
import numpy as np
from PIL import Image
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

TRAIN_PATH = './train'
EXTENSION = '.jpg'

#CNN model for our training
#convolutional layer -> ReLU -> pooling layer -> convolutional layer -> ReLU -> pooling
#-> flatten -> fully connected -> ReLU -> fully connected -> ReLU -> hidden fully connected output
class Model_PA3(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 20, 5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(20, 40, 5, stride=1, padding=1)
        self.fc1 = nn.Linear(40 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 100)
        self.output_fc = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #flatten the convolutional to a dimension of 40 * 6 * 6
        x = x.view(-1, 40 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        features = self.output_fc(x)
        return features

#main function where we construct our model, load the data, train it and save it
def main():
    #our classes
    # classes = ('diving', 'golf swing', 'kicking', 'lifting', 'riding horse', 'running', 'skateboarding', 'swing bench')
    #instantiate our model, using cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model_PA3().to(device)
    #Construct dataset, call data loader
    #get the training dataset
    print()
    print('Generating training dataset')
    train_dataset = data_loader()

    #train our model, save it, then test it with another script
    print('Training model using 20 epochs')
    for epoch in range(20):
        trainer(model, train_dataset, epoch, device)
    #save our model
    print('Saving our model')
    print()
    torch.save(model.state_dict(), "pa3_cnn.pt")
    torch.save(model.state_dict(), "pa3_cnn.pth")
   

#function to train our model
def trainer(model, trainset, epoch, device):
    #put our model in training mode, needed for fitting and saving our model
    model.train()
    #other loss functions I considered
    #nll loss - subset of cross entropy
    #smooth l1 - less sensitive to outliers 
    #cross entropy - hihg pentalty when highly confident incorrect prediction
    #leibler divergence - more computing power needed than cross entropy, similar results
    #SGD optimzer from pa2, using the same SGD momentum and learning rate did not work
    # optimizer = optim.SGD(model.parameters(), lr=.003, momentum=.9, weight_decay=.00001)
    #decided to use the Adam optimizer and reduced the learning rate to handle the larger dataset, same decay
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=.00001)
    #initialize our accuracy and loss
    accuracy = 0.0
    #enumerate through our training data
    for batch_idx, (data, target) in  enumerate(trainset):
        #store on memory if cuda is available
        inputs = data.to(device)
        labels = target.to(device)
        #zero the gradient
        optimizer.zero_grad()
        #call our model with inputs to get features
        features = model(inputs)
        #grab our detected features
        loss = F.cross_entropy(features, labels)
        #back propagate the loss function
        loss.backward()
        #weight updates with optimzer
        optimizer.step()
        #print statement with formatting from pa2
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(inputs), len(trainset.dataset),
                       100. * batch_idx / len(trainset), loss.item()))


#function to load our training data
def data_loader():
    #transformation for our image
    #resize, convert to tensor, normalize data
    dataset_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((32,32)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,.5,.5), (0.5,.5,.5))
    ])

    #load the dataset and transform to tensors
    #training dataset
    data_train = torchvision.datasets.ImageFolder(TRAIN_PATH, transform=dataset_transform)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=100, shuffle=True,num_workers=1)
    return train_loader
    


if __name__ == "__main__":
    main()