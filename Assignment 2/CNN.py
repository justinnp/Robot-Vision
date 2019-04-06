# __author__ = "Aisha Urooj"

# Justin Powell
# UCF  - CAP4453
# Spring 2019 

import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        # ======================================================================
        # One fully connected layer.
        # Not working: self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.Sigmoid())

        #Forward process of our Model 1
        #28x28 -> first fully connected layer -> 100 -> second fully connected layer -> 10 features

        #fully connected layer, input of 28x28, outputting 100 features
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features 
        # wrap our first fully connected layer in a sigmoid fucntion   
        x = F.sigmoid(self.fc1(x))
        #pass that output of the first fully connected layer into the output layer
        features = self.output_layer(x)
        return  features

class Model_2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #Forward process of our Model 1
        #28x28 -> first convolutional layer -> 20 features -> second convolutional layer -> 40 features ->
        #flatten 40 features * shape=(4,4) -> first fc layer -> 100 -> second fc layer -> 10 features

        #convolutional layer, input is 1, output features are 20, kernel size of 5 and a stride of 1
        self.conv1 = nn.Conv2d(1,20,5,stride=1)
        self.conv2 = nn.Conv2d(20,40,5,stride=1)
        self.fc1 = nn.Linear(40*4*4, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    
        #add a pooling layer to our convolutional layers with a stride of 2
        x = F.max_pool2d(self.conv1(x),2)
        x = F.max_pool2d(self.conv2(x),2)
        #flatten to 640 since the input of the first linear layer is 4 * 4 * 40 which = 640
        x = x.view(-1, 640)
        # wrap our first fully connected layer in a sigmoid fucntion   
        x = F.sigmoid(self.fc1(x))
        features = self.output_layer(x)
        return  features


class Model_3(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #Same forward process as our model 2
        self.conv1 = nn.Conv2d(1,20,5,stride=1)
        self.conv2 = nn.Conv2d(20,40,5,stride=1)
        #input is the output of the previous conv layer which was [batch_size, 40, 4, 4]
        self.fc1 = nn.Linear(40*4*4, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features  
        #wrap our convolutional layers in a ReLu
        x = F.relu(self.conv1(x))
        #pooling layer of size 2 with stride of 2
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        #flatten to 640 since the input of the first linear layer is 4 * 4 * 40 which = 640
        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        features = self.output_layer(x)
        return  features

class Model_4(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #28x28 -> first convolutional layer -> 20 features -> second convolutional layer -> 40 features ->
        #flatten 40 features * shape=(4,4) -> first fc layer -> 100 -> second fc layer -> 100 features ->
        #final output layer -> 10 features
        self.conv1 = nn.Conv2d(1,20,5,stride=1)
        self.conv2 = nn.Conv2d(20,40,5,stride=1)
        self.fc1 = nn.Linear(40*4*4,100)
        self.fc2 = nn.Linear(100,hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    
        #Forward process works the same as the other models, except this has an added fc layer
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        #flatten to 640 since the input of the first linear layer is 4 * 4 * 40 which = 640
        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        #our second fc layer, pass the output of the first fc layer into the second
        x = F.relu(self.fc2(x))
        features = self.output_layer(x)
        return  features

class Model_5(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.

        #28x28 -> first convolutional layer -> 20 features -> second convolutional layer -> 40 features ->
        #flatten 40 features * shape=(4,4) -> first fc layer -> 1000 -> second fc layer -> 1000 features ->
        #final output layer -> 10 features
        self.conv1 = nn.Conv2d(1,20,5,stride=1)
        self.conv2 = nn.Conv2d(20,40,5,stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(40*4*4,1000)
        self.fc2 = nn.Linear(1000,hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features   #
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        #flatten to 640 since the input of the first linear layer is 4 * 4 * 40 which = 640
        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #pass the second fc layer through a drop out of .5 to aid in preventing overfitting
        x = self.dropout(x)
        features = self.output_layer(x)
        return  features


class Net(nn.Module):
    def __init__(self, mode, args):
        super().__init__()
        self.mode = mode
        self.hidden_size= args.hidden_size
        # model 1: base line
        if mode == 1:
            in_dim = 28*28 # input image size is 28x28
            self.model = Model_1(in_dim, self.hidden_size)

        # model 2: use two convolutional layer
        if mode == 2:
            self.model = Model_2(self.hidden_size)

        # model 3: replace sigmoid with relu
        if mode == 3:
            self.model = Model_3(self.hidden_size)

        # model 4: add one extra fully connected layer
        if mode == 4:
            self.model = Model_4(self.hidden_size)

        # model 5: utilize dropout
        if mode == 5:
            self.model = Model_5(self.hidden_size)


    def forward(self, x):
        if self.mode == 1:
            x = x.view(-1, 28* 28)
            x = self.model(x)
        if self.mode in [2, 3, 4, 5]:
            x = self.model(x)
        # ======================================================================
        # Define softmax layer, use the features.
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign calculated value to logits after code implementation.
        logits = F.softmax(x, dim=0)
        return logits

