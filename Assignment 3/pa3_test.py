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

TEST_PATH = './test'
EXTENSION = '.jpg'
MODEL_PATH = './models/pa3_cnn.pt'

#main function where we load our model, construct testing dataset, test the model and show accuracy
def main():
    #our classes
    classes = ('diving', 'golf swing', 'kicking', 'lifting', 'riding horse', 'running', 'skateboarding', 'swing bench')
    #load our model
    model = Model_PA3()
    model.load_state_dict(torch.load(MODEL_PATH))
    #put model in evaluation mode for testing
    model.eval()
    #Construct datasets, call data loader
    #get the training dataset
    print()
    print('Generating testing dataset')
    test_dataset = data_loader()
    #getting overall accuracy of our model
    print('Running dataset through model')
    acc = tester(model, test_dataset)
    print('Accuracy of entire dataset is ' + str(acc))
    print()
    #get accuracy of random images from the dataset
    print('Pulling random images from dataset')
    it = iter(test_dataset)
    images, labels = it.next()
    accur = tester(model, it)
    print('Accuracy of random images from dataset is ' + str(accur))
    print()


#function to test our data against our model
def tester(model, testset):
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(testset):
        inputs = data
        labels = target
        #get our features from the model
        features = model(inputs)
        #get max of our feature vector, which is our best prediction
        tens, prediction = torch.max(features.data, 1)
        total += labels.size(0)
        #test prediction against actual label
        correct += (prediction == labels).sum().item()
    acc = (correct / total) * 100
    return acc 


#function to load our testing data
def data_loader():
    #transformation for our image
    #resize, convert to tensor, normalize data
    dataset_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((32,32)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,.5,.5), (0.5,.5,.5))
    ])
    #testing dataset
    data_test = torchvision.datasets.ImageFolder(TEST_PATH, transform=dataset_transform)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=50, shuffle=False, num_workers=1)
    return test_loader

#needed for model loading
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
        x = x.view(-1, 40 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        features = self.output_fc(x)
        return features

if __name__ == "__main__":
    main()