from matplotlib import patches
import numpy as np
import os
import math
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.io import read_image
import torchvision.transforms.functional as fn
import matplotlib.pyplot as plt


def crop_image(image, left_landmarks, right_landmarks, max_size, max_x_y):
    # Crop the image to the bounding box from the landmark of the left and right eye
    tlx, tly, brx, bry = left_landmarks
    tlx = 448 - tlx
    brx = 448 - brx
    if abs(brx - tlx) * abs(bry - tly) < max_size:
        tlx = int(tlx + (max_x_y[0] - abs(brx - tlx)))
        tly = int(tly - (max_x_y[1] - abs(bry - tly)))
        brx = int(brx - (max_x_y[0] - abs(brx - tlx)))
        bry = int(bry + (max_x_y[1] - abs(bry - tly)))

        if abs(brx - tlx) * abs(bry - tly) != max_size:
            print([abs(brx - tlx), abs(bry - tly)])
            print(abs(brx - tlx) * abs(bry - tly))
            print(max_x_y)
            print(max_size)
            raise Exception(
                "The cropped image is not the same size as the max size")

    cropped_left = fn.crop(image, tly, brx, abs(bry - tly), abs(brx - tlx))
    # padded_left = fn.center_crop(cropped_left, (153, 95))

    tlx, tly, brx, bry = right_landmarks

    tlx = 448 - tlx
    brx = 448 - brx
    if abs(brx - tlx) * abs(bry - tly) < max_size:
        tlx = int(tlx + (max_x_y[0] - abs(brx - tlx)))
        tly = int(tly - (max_x_y[1] - abs(bry - tly)))
        brx = int(brx - (max_x_y[0] - abs(brx - tlx)))
        bry = int(bry + (max_x_y[1] - abs(bry - tly)))

        if abs(brx - tlx) * abs(bry - tly) != max_size:
            print([abs(brx - tlx), abs(bry - tly)])
            print(abs(brx - tlx) * abs(bry - tly))
            print(max_x_y)
            print(max_size)
            raise Exception(
                "The cropped image is not the same size as the max size")

    cropped_right = fn.crop(image, tly, brx, abs(bry - tly), abs(brx - tlx))
    # padded_right = fn.center_crop(cropped_right, (153, 95))

    return cropped_left, cropped_right
    # return padded_left, padded_right


def process_image(image):
    # Perform a histogram equalization and grayscaling on the tensor image
    equalized = fn.equalize(image.contiguous())
    return equalized


def find_max_landmark(left_landmarks, right_landmarks):
    return 13616, [148, 92]  # Hardcoded because we know the max size

    landmarks = np.concatenate((left_landmarks, right_landmarks), axis=0)

    max_size = 0
    max_x_y = [0,  0]

    for i in range(len(landmarks)):
        tlx, tly, brx, bry = landmarks[i]
        size = abs(brx - tlx) * abs(bry - tly)

        if size > max_size:
            max_size = size
            max_x_y = [abs(brx - tlx), abs(bry - tly)]

    return max_size, max_x_y


def create_dataset_from_mix(path):
    path = os.path.join(path, 'mix')

    pairs = []

    # load labels and landmarks
    labels = np.load(os.path.join(path, 'labels.npy'))
    left_landmarks = np.load(os.path.join(path, 'left_landmark_mix.npy'))
    right_landmarks = np.load(os.path.join(path, 'right_landmark_mix.npy'))

    max_size, max_x_y = find_max_landmark(left_landmarks, right_landmarks)

    # load images from path
    print("Loading & preprocessing dataset 🚀")
    for i in tqdm(range(10)):
        img_path = os.path.join(path, f'{i}.png')
        img = read_image(img_path)

        # Crop the image
        cropped_left, cropped_right = crop_image(
            img, left_landmarks[i], right_landmarks[i], max_size, max_x_y)

        # Process the image
        processed_left = process_image(cropped_left)
        processed_right = process_image(cropped_right)

        # Append the processed image and label to their respective dataset
        pairs.append([(processed_left, labels[i]),
                     (processed_right, labels[i])])
    return pairs


class ColumbiaGaze(Dataset):
    def __init__(self, path):
        data_set_left, data_set_right = create_dataset_from_mix(path)
        self.eyes = data_set_left + data_set_right

    def __len__(self):
        return len(self.eyes)

    def __getitem__(self, idx):
        return self.eyes[idx]


def create_random_pair(eyes, add_extra_pair):
    random_list = []
    # Get a random permutation of the indices
    indices = torch.randperm(len(eyes))
    for i in range(0, len(indices)-1, 2):
        # Append two eyes
        random_list.append([eyes[indices[i]], eyes[indices[i+1]]])

    if add_extra_pair:
        random_index = torch.randint(0, len(eyes)-1, (1, 1))
        random_list.append([eyes[indices[-1]], eyes[random_index]])

    return random_list


def create_eye_pair_dataset(path):
    data_set = []

    # 2 directories are eye_landmark and mix
    num_ids = len(os.listdir(path)) - 2

    # For each participant crop and process the images and append that list to the dataset
    print("Loading & preprocessing dataset 🚀")
    for i in tqdm(range(1, num_ids)):
        participant_path = os.path.join(path, str(i))

        participant_eyes_left = []
        participant_eyes_right = []

        labels = np.load(os.path.join(participant_path, 'labels.npy'))
        left_landmarks = np.load(os.path.join(
            participant_path, 'left_landmarks.npy'))
        right_landmarks = np.load(os.path.join(
            participant_path, 'right_landmarks.npy'))

        max_size, max_x_y = find_max_landmark(left_landmarks, right_landmarks)
        # 3 files are labels, left_landmarks and right_landmarks
        num_imgs = len(os.listdir(participant_path)) - 3
        for j in range(num_imgs):
            img_path = os.path.join(participant_path, f'{j}.png')
            img = read_image(img_path)

            # Crop the image
            cropped_left, cropped_right = crop_image(
                img, left_landmarks[j], right_landmarks[j], max_size, max_x_y)

            # Process the image
            processed_left = process_image(cropped_left)
            processed_right = process_image(cropped_right)

            participant_eyes_left.append((processed_left, labels[j]))
            participant_eyes_right.append((processed_right, labels[j]))

        # each participant has 105 images so for each participant needs to have 52 left and 53 pairs or the other way around
        # the i % 2 makes sure that when i is even a extra left pair is adden and when i is odd and extra right pair is added
        random_left = create_random_pair(participant_eyes_left, i % 2 == 0)
        random_right = create_random_pair(participant_eyes_right, i % 2 == 1)

        data_set += random_left
        data_set += random_right

    return data_set


def show_eye_pair_dataset():
    data_set = create_eye_pair_dataset('data')
    fig, axs = plt.subplots(nrows=10, ncols=2, squeeze=False)
    for i in range(10):
        img1 = data_set[i][0][0]
        img2 = data_set[i][1][0]

        axs[i, 0].imshow(img1.permute(1, 2, 0))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[i, 1].imshow(img2.permute(1, 2, 0))
        axs[i, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


def show_gaze_pair_dataset():
    data_set = create_dataset_from_mix('data')
    fig, axs = plt.subplots(nrows=10, ncols=2, squeeze=False)
    for i in range(0, 10, 2):
        img1 = data_set[i][0][0]
        img2 = data_set[i][1][0]

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

# gaze similar
# [ [linkeroog kandidaat 1 foto 1, rechteroog kandidaat 1 foto 1]]

# eye similar
# [ [random linker oog kandiaat 1, random ander linker oog kandidaat 1], [] ]


class ColumbiaEye(Dataset):
    def __init__(self, path):
        self.eyes = create_eye_pair_dataset(path)

    def __len__(self):
        return len(self.eyes)

    def __getitem__(self, idx):
        return self.eyes[idx]


class ColumbiaEyeGaze(Dataset):
    def __init__(self, path):
        self.gaze_pairs = create_dataset_from_mix(path)
        self.eye_pairs = create_eye_pair_dataset(path)

        # each entry is an array of 4 image label tuples
        #   - indices 0 and 1 contain the gaze pair
        #   - indices 2 and 3 contain the eye pair
        # example [(img, label), (img, label), (img, label), (img, label)]
        self.data = [self.gaze_pairs[i] + self.eye_pairs[i]
                     for i in range(len(self.gaze_pairs))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
