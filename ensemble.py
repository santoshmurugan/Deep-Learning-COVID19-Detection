import torch
import torch.nn as nn
import torchvision.models as models

class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()

        self.model1 = models.alexnet(pretrained=True)
        self.model1.classifier[6] = nn.Linear(4096,2)
        
        self.model2 = models.vgg11_bn(pretrained=True)
        self.model2.classifier[6] = nn.Linear(4096,2)

        self.model3 = models.resnet18(pretrained = True)
        self.model3.fc = nn.Linear(512,2)

        self.linear = nn.Linear(6, 2)

    def forward(self, x):
        output = torch.cat((self.model1(x), self.model2(x), self.model3(x)), dim=0)
        output = torch.flatten(output)
        output = self.linear(output)
        return output