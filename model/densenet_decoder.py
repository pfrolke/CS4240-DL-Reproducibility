from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.densenet import DenseNetBlock, DenseNetTransitionUp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseNetDecoder(nn.Module):

    def __init__(self, c_in, activation_fn, normalization_fn,
                 ):
        super(DenseNetDecoder, self).__init__()

        num_blocks = 4
        num_layers_per_block = 4
        growth_rate = 32

        c_to_concat = [0] * (num_blocks + 2)

        c_now = c_in
        for i in range(num_blocks):
            i_ = i + 1
            # Define dense block
            self.add_module('block%d' % i_, DenseNetBlock(
                c_now,
                num_layers_per_block,
                growth_rate,
                activation_fn,
                normalization_fn,
            ))
            c_now = list(self.children())[-1].c_now

            # Define transition block if not last layer
            if i < (num_blocks - 1):
                self.add_module('trans%d' % i_, DenseNetTransitionUp(
                    c_now,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                ))
                c_now = list(self.children())[-1].c_now
                c_now += c_to_concat[i]

        # Last up-sampling conv layers
        self.last = DenseNetDecoderLastLayers(c_now,
                                              growth_rate,
                                              activation_fn,
                                              normalization_fn,
                                              skip_connection_growth=c_to_concat[-1])
        self.c_now = 1

    def forward(self, x):
        # Apply initial layers and dense blocks
        for module in self.children():
            x = module.forward(x)
        return x


class DenseNetDecoderLastLayers(nn.Module):

    def __init__(self, c_in, growth_rate, activation_fn, normalization_fn, skip_connection_growth):
        super(DenseNetDecoderLastLayers, self).__init__()
        # First deconv
        self.conv1 = nn.ConvTranspose2d(c_in, 4 * growth_rate, bias=False,
                                        kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        nn.init.kaiming_normal_(self.conv1.weight.data)

        # Second deconv
        c_in = 4 * growth_rate + skip_connection_growth
        self.norm2 = normalization_fn(
            c_in, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)
        self.conv2 = nn.ConvTranspose2d(c_in, 2 * growth_rate, bias=False,
                                        kernel_size=1, stride=1, padding=0,
                                        output_padding=0)
        nn.init.kaiming_normal_(self.conv2.weight.data)

        # Final conv
        c_in = 2 * growth_rate
        c_out = 1
        self.norm3 = normalization_fn(
            c_in, track_running_stats=False).to(device)
        self.conv3 = nn.Conv2d(c_in, c_out, bias=False,
                               kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv3.weight.data)
        self.c_now = c_out

    def forward(self, x):
        x = self.conv1(x)
        #
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        #
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv3(x)
        return x
