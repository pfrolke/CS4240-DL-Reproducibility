import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, densenet121, DenseNet121_Weights
from torchvision.io import read_image
import torchvision.transforms.functional as fn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CrossEncoder(nn.Module):
    def __init__(self, gaze_dim=15, eye_dim=32):
        super().__init__()

        # encoder
        self.transforms = ResNet18_Weights.DEFAULT.transforms()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

        # shared + specific feature layer
        z_dim = gaze_dim + eye_dim
        self.encoder_to_gaze_and_eye = nn.Linear(
            self.encoder.fc.out_features, z_dim)

        # feed into decoder
        self.gaze_and_eye_to_decoder = nn.Linear(z_dim, )

        # decoder
        self.decoder = densenet121(
            weights=DenseNet121_Weights.DEFAULT).eval()  # todo ???
