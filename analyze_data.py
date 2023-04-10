from matplotlib import patches
import numpy as np
from torchvision.io import read_image
import matplotlib.pyplot as plt
from load_data import create_pair_dataset, process_image


def show_eye_pair_dataset():
    _, eye_pair = create_pair_dataset('data')
    _, axs = plt.subplots(nrows=10, ncols=2, squeeze=False)
    for i in range(10):
        img1 = eye_pair[i][0][0]
        img2 = eye_pair[i][1][0]

        axs[i, 0].imshow(img1.permute(1, 2, 0))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[i, 1].imshow(img2.permute(1, 2, 0))
        axs[i, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


def show_gaze_pair_dataset():
    gaze_pair, _ = create_pair_dataset('data')
    _, axs = plt.subplots(nrows=10, ncols=2, squeeze=False)
    for i in range(0, 10, 2):
        img1 = gaze_pair[i][0][0]
        img2 = gaze_pair[i][1][0]

        axs[i, 0].imshow(img1.permute(1, 2, 0))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[i, 1].imshow(img2.permute(1, 2, 0))
        axs[i, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


def show_one_image():
    left_landmarks = np.load('data/5/left_landmarks.npy')
    right_landmarks = np.load('data/5/right_landmarks.npy')

    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False)
    for i in range(1):
        img = read_image(f'data/5/{i+95}.png')

        tlx, tly, brx, bry = left_landmarks[i+95]
        tlx = 448 - tlx
        brx = 448 - brx

        # Display the image
        axs[i, 0].imshow(img.permute(1, 2, 0))

        # Create a Rectangle patch
        rect = patches.Rectangle((brx, tly), abs(
            brx - tlx), abs(bry - tly), linewidth=1, edgecolor='r', facecolor='none')
        axs[i, 0].add_patch(rect)

        tlx, tly, brx, bry = right_landmarks[i+95]
        tlx = 448 - tlx
        brx = 448 - brx
        rect = patches.Rectangle((brx, tly), abs(
            brx - tlx), abs(bry - tly), linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        axs[i, 0].add_patch(rect)
    plt.show()


show_eye_pair_dataset()
