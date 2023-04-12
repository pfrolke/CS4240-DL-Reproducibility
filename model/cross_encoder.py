import torch
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

        # shared + specific feature layer
        # channels
        self.gaze_and_eye_dim = gaze_dim + eye_dim
        self.encoder_to_gaze_and_eye = nn.Linear(
            self.encoder.fc.out_features, self.gaze_and_eye_dim)

        # feed into decoder
        self.gaze_and_eye_to_decoder = nn.Linear(
            self.gaze_and_eye_dim, decoder_input_c)

        # decoder
        self.decoder = DenseNetDecoder(
            decoder_input_c,
            activation_fn=nn.LeakyReLU,
            normalization_fn=nn.InstanceNorm2d,
        )

    def forward(self, data):
        # data shape: (batch_size, n_img, img_colors, img_height, img_width)
        # n_img = 4: gaze_pair + eye_pair

        batch_size, n_img, img_colors, img_height, img_width = data.shape

        assert n_img == 4

        # stack all images
        # x[0] = data[0][0]
        # x[4] = data[1][0]
        x = data.view(batch_size * n_img, img_colors, img_height, img_width)

        # encoder
        x = self.encoder(x.float())
        x = self.encoder_to_gaze_and_eye(x)
        x = x.view(batch_size, n_img, self.gaze_and_eye_dim)

        # x.shape = (batch_size, n_img, gaze_dim + eye_dim)
        x_gaze = x[:, :, :self.gaze_dim]
        x_eye = x[:, :, self.gaze_dim:]

        # swap gaze for indices 0 and 1
        x_gaze_swapped = torch.clone(x_gaze)
        x_gaze_swapped[:, 0, :] = x_gaze[:, 1, :]
        x_gaze_swapped[:, 1, :] = x_gaze[:, 0, :]

        # swap eye for indices 2 and 3
        x_eye_swapped = torch.clone(x_eye)
        x_eye_swapped[:, 2, :] = x_eye[:, 3, :]
        x_eye_swapped[:, 3, :] = x_eye[:, 2, :]

        # recombine swapped tensors
        x = torch.cat((x_gaze_swapped, x_eye_swapped), dim=2)

        # return to stacked view
        x = x.view(batch_size * n_img, 1, self.gaze_and_eye_dim)

        # decoder
        x = self.gaze_and_eye_to_decoder(x)
        x = self.decoder(x)

        return x
