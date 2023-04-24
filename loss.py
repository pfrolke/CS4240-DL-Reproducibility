import torch.nn as nn


def loss_fn(eye_pairs, outputs):
    # data.shape = (64, 1, 32, 64)
    # outputs.shape = (64, 1, 32, 64)

    eye_pairs = eye_pairs.view(-1, 4, 32, 64)
    outputs = outputs.view(-1, 4, 32, 64)

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
    return l1_loss_fn(eye_pairs[:, index, :, :], outputs[:, index, :, :])


def residual_loss(eye_pairs, outputs, index1, index2):
    l1_loss_fn = nn.L1Loss(reduction='mean')
    diff_eye1 = eye_pairs[:, index1, :, :] - outputs[:, index1, :, :]
    diff_eye2 = eye_pairs[:, index2, :, :] - outputs[:, index2, :, :]
    return l1_loss_fn(diff_eye1, diff_eye2)
