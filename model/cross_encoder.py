import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from model.densenet_decoder import DenseNetDecoder


class CrossEncoder(nn.Module):
    def __init__(self, gaze_dim=15, eye_dim=32, decoder_input_c=16):
        super().__init__()
        self.gaze_dim = gaze_dim
        self.eye_dim = eye_dim

        # encoder
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

        # (shared + specific feature layer) * 3 color channels
        gaze_and_eye_dim = (gaze_dim + eye_dim) * 3
        self.encoder_to_gaze_and_eye = nn.Linear(
            self.encoder.fc.out_features, gaze_and_eye_dim)

        # feed into decoder
        self.gaze_and_eye_to_decoder = nn.Linear(
            gaze_and_eye_dim, decoder_input_c)

        # decoder
        self.decoder = DenseNetDecoder(
            decoder_input_c,
            activation_fn=nn.LeakyReLU,
            normalization_fn=nn.InstanceNorm2d,
        )

    def forward(self, data):
        # x: batch_size * [(img1, label), (img2, label), (img3, label), (img4, label)]

        x = self.encoder(x)
        x = self.encoder_to_gaze_and_eye(x)
        x = self.gaze_and_eye_to_decoder(x)
        x = self.decoder(x)

        return x
