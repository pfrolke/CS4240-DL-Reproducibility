import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class GazeEstimatorSupervised(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = ResNet18_Weights.DEFAULT.transforms()
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        self.linear = nn.Linear(self.resnet18.fc.out_features, 2)

    def forward(self, data):
        x = self.transforms(data)
        x = self.resnet18(x)
        return self.linear(x)
