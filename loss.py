from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from load_data import ColumbiaPairs
from model.cross_encoder import CrossEncoder


def loss_fn(eye_pairs, outputs):
    # assuming the input and output will both be [eye1, eye2, eye3, eye4] * batch_size

    alpha = 0.5
    beta = 1.0
    gamma = 0.5

    loss_eye1 = l1_loss(eye_pairs, outputs, 0)
    loss_eye2 = l1_loss(eye_pairs, outputs, 1)
    residual_loss_eye_pair = residual_loss(eye_pairs, outputs, 0, 1)

    loss_eye_pair = loss_eye1 + loss_eye2 + alpha * residual_loss_eye_pair

    loss_eye3 = l1_loss(eye_pairs, outputs, 2)
    loss_eye4 = l1_loss(eye_pairs, outputs, 3)
    residual_loss_gaze_pair = residual_loss(eye_pairs, outputs, 2, 3)

    loss_gaze_pair = loss_eye3 + loss_eye4 + gamma * residual_loss_gaze_pair

    final_loss = loss_eye_pair + beta * loss_gaze_pair

    return final_loss.data.clone().detach().requires_grad_(True)


def l1_loss(eye_pairs, outputs, index):
    l1_loss_fn = nn.L1Loss(reduction='mean')
    loss_per_instance = l1_loss_fn(
        eye_pairs[:, :, index], outputs[:, :, index])
    return torch.stack(loss_per_instance).sum(dim=0)


def residual_loss(eye_pairs, outputs, index1, index2):
    l1_loss_fn = nn.L1Loss(reduction='mean')
    res_loss_per_instance = [l1_loss_fn(
        eyes[index1] - output[index1], eyes[index2] - output[index2]) for eyes, output in zip(eye_pairs, outputs)]
    return torch.stack(res_loss_per_instance).sum(dim=0)
