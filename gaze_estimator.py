import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.io import read_image
import torchvision.transforms.functional as fn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SupervisedGazeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = ResNet18_Weights.DEFAULT.transforms()
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        self.linear = nn.Linear(self.resnet18.fc.out_features, 2)

    def forward(self, data):
        x = self.transforms(x)
        x = self.resnet18(data)
        return self.linear(x)
