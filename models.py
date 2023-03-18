from torchvision import datasets, models, transforms
import torch.nn as nn

class ResNet18_Scratch(nn.Module):
    def __init__(self,dropout):
        super(ResNet18_Scratch, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        
        # Here the size of each output sample is set to 2.
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 2)
        )


    def forward(self, x):
        return self.model(x)
        



class ResNet18(nn.Module):
    def __init__(self, dropout):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        # Here the size of each output sample is set to 2.
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 2)
        )


    def forward(self, x):
        return self.model(x)

