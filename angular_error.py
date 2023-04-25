import math
import torch


def spherical2cartesial(x):
    # angle to radian for Columbia
    x = x*math.pi/180

    output = torch.zeros(x.size(0), 3).cuda()
    output[:, 2] = torch.cos(x[:, 0])*torch.cos(x[:, 1])
    output[:, 0] = torch.cos(x[:, 0])*torch.sin(x[:, 1])
    output[:, 1] = torch.sin(x[:, 0])

    return output


def compute_angular_error(input, target):
    input = spherical2cartesial(input)
    target = spherical2cartesial(target)

    input = input.view(-1, 3, 1)
    target = target.view(-1, 1, 3)
    output_dot = torch.bmm(target, input)
    # print(output_dot)
    output_dot = output_dot.view(-1)
    output_dot = output_dot.clamp(-1., 1.)
    output_dot = torch.acos(output_dot)

    output_dot = output_dot.data
    output_dot = 180*torch.sum(output_dot)/math.pi
    return output_dot
