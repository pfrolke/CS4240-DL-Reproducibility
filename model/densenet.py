import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DenseNetBlock(nn.Module):

    def __init__(self, c_in, num_layers, growth_rate, activation_fn, normalization_fn):
        super(DenseNetBlock, self).__init__()
        c_now = c_in
        kernel_size = 3
        for i in range(num_layers):
            i_ = i + 1
            self.add_module('compo%d' % i_, DenseNetCompositeLayer(
                c_now, growth_rate, kernel_size, activation_fn, normalization_fn
            ))
            c_now += growth_rate
        self.c_now = c_now

    def forward(self, x):
        for module in self.children():
            x_before = x
            x = module.forward(x)
            x = torch.cat([x_before, x], dim=1)
        return x


class DenseNetCompositeLayer(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, activation_fn, normalization_fn):
        super(DenseNetCompositeLayer, self).__init__()
        self.norm = normalization_fn(c_in).to(device)
        self.act = activation_fn(inplace=True)
        self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size, stride=1,
                                       padding=1, bias=False).to(device)
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.c_now = c_out

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class DenseNetTransitionUp(nn.Module):

    def __init__(self, c_in, activation_fn, normalization_fn):
        super(DenseNetTransitionUp, self).__init__()
        c_out = c_in
        self.norm = normalization_fn(
            c_in, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)
        self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=3,
                                       stride=2, padding=1, output_padding=1,
                                       bias=False).to(device)
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.c_now = c_out

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x
