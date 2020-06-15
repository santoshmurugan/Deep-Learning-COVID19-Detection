import torch
import torch.nn as nn
import torchvision.models as models

class NovelNet(nn.Module):   
    def __init__(self):
        super(NovelNet, self).__init__() #Boilerplate from https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, stride = 3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(128,256, kernel_size = 7, padding = 2),
            nn.ReLU(inplace = True),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.linear = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(4096, 1028),
            nn.LeakyReLU(inplace =True),
            nn.Dropout(p=0.2),
            nn.Linear(1028, 256),
            nn.Tanh(),
            nn.Dropout(p = 0.2),
            nn.Linear(256,2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x,1) #Might run into dimension issue here
        x = self.linear(x)
        return x