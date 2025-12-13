import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN,self).__init__()
        self.features = nn.Sequential(
            # (c,h,w) =(3,244,244)
            nn.Conv2d(3,16,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out 122 x 122 x 16

            nn.Conv2d(16,32,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out 61 x 61 x 32

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # out 30 x 30 x 64
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*30*30, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

