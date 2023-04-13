import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from model.densenet_decoder import DenseNetDecoder


class CrossEncoder(nn.Module):
    def __init__(self, gaze_dim=15, eye_dim=32, decoder_input_c=16):
        super().__init__()
        self.gaze_dim = gaze_dim
        self.eye_dim = eye_dim
        self.decoder_input_c = decoder_input_c

        bottleneck_shape = (2, 4)
        self.bottleneck_shape = bottleneck_shape
        enc_num_all = np.prod(bottleneck_shape) * self.decoder_input_c

        # encoder
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

        # shared + specific feature layer
        # channels
        self.gaze_and_eye_dim = 3 * gaze_dim + eye_dim
        self.encoder_to_gaze_and_eye = self.linear(
            self.encoder.fc.out_features, self.gaze_and_eye_dim)

        # feed into decoder
        self.gaze_and_eye_to_decoder = self.linear(
            self.gaze_and_eye_dim, enc_num_all)

        # decoder
        self.decoder = DenseNetDecoder(
            decoder_input_c,
            activation_fn=nn.LeakyReLU,
            normalization_fn=nn.InstanceNorm2d,
        )

    def linear(self, f_in, f_out):
        fc = nn.Linear(f_in, f_out)
        nn.init.kaiming_normal_(fc.weight.data)
        nn.init.constant_(fc.bias.data, val=0)
        return fc

    def forward(self, data):
        # data shape: (batch_size, n_img, img_colors, img_height, img_width)
        # n_img = 4: gaze_pair + eye_pair

        batch_size, n_img, img_height, img_width = data.shape

        assert n_img == 4

        # stack all images
        # x[0] = data[0][0]
        # x[4] = data[1][0]
        x = data.view(batch_size * n_img, 1, img_height, img_width)

        x = x.expand(batch_size * n_img, 3, img_height, img_width)

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
        # x.shape = (16, 4, 47) = (batch_size, num_img, gaze_dim+eye_dim)

        # return to stacked view
        x = x.view(batch_size * n_img, self.gaze_and_eye_dim)
        # x.shape = (64, 47) = (batch_size*num_img, 1, gaze_dim+eye_dim)

        # decoder
        x = self.gaze_and_eye_to_decoder(x) 
        # x.shape = (64, 16) = (batch_size*num_img, dec_in)

        x = x.view(batch_size*n_img, self.decoder_input_c, *self.bottleneck_shape) 
        # x.shape = (64, 16, 2, 4) = (batch_size*num_img, dec_in, (bottleneck))

        x = self.decoder(x)
        # x.shape = (64, 1, 32, 64)

        x = x.view(batch_size, n_img, 32, 64)

        return x
