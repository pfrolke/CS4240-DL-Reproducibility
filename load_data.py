from matplotlib import patches
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.io import read_image
import torchvision.transforms.functional as fn
import matplotlib.pyplot as plt


def crop_image(image, landmarks, width, height):
    '''Crop the image centered on the bounding box from the landmark with the width and height given'''

    tlx, tly, brx, bry = landmarks

    # offset difference to required width and height
    tlx -= (width - abs(brx - tlx))
    brx += (width - abs(brx - tlx))
    tly -= (height - abs(bry - tly))
    bry += (height - abs(bry - tly))

    # size should be equal to required size
    assert abs(brx - tlx) * abs(bry - tly) == width * height
    assert abs(brx - tlx) == width
    assert abs(bry - tly) == height

    # horizontally flip landmarks since they were generated that way
    return fn.crop(image, tly, 448 - brx, height, width)


def process_image(image):
    '''Histogram equalization and grayscaling on the tensor image'''
    equalized = fn.equalize(image.contiguous())
    return equalized


def find_max_landmark(path):
    '''Find maximum landmark width and height'''
    num_participants = len(os.listdir(path)) - 2

    max_width = -1
    max_height = -1

    for i in range(1, num_participants):
        participant_path = os.path.join(path, str(i))

        # load labels and landmarks
        left_landmarks = np.load(os.path.join(
            participant_path, 'left_landmarks.npy'))
        right_landmarks = np.load(os.path.join(
            participant_path, 'right_landmarks.npy'))

        all_landmarks = np.concatenate(
            (left_landmarks, right_landmarks), axis=0)

        for tlx, tly, brx, bry in all_landmarks:
            max_width = max(abs(brx - tlx), max_width)
            max_height = max(abs(bry - tly), max_height)

    return max_width, max_height


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


def create_pair_dataset(path):
    eye_pairs = []
    gaze_pairs = []

    # 2 directories are eye_landmark and mix
    num_participants = len(os.listdir(path)) - 2

    # For each participant crop and process the images and append that list to the dataset
    print("Loading & preprocessing dataset 🚀")
    max_width, max_height = find_max_landmark(path)

    for i in tqdm(range(1, num_participants)):
        participant_path = os.path.join(path, str(i))

        participant_eyes_left = []
        participant_eyes_right = []

        # load labels and landmarks
        labels = np.load(os.path.join(participant_path, 'labels.npy'))
        left_landmarks = np.load(os.path.join(
            participant_path, 'left_landmarks.npy'))
        right_landmarks = np.load(os.path.join(
            participant_path, 'right_landmarks.npy'))

        # loop over all images per subject
        for j in range(labels.shape[0]):
            img = read_image(os.path.join(participant_path, f'{j}.png'))

            # Crop the image
            cropped_left = crop_image(
                img, left_landmarks[j], max_width, max_height)
            cropped_right = crop_image(
                img, right_landmarks[j], max_width, max_height)

            # Process the image
            processed_left = process_image(cropped_left)
            processed_right = process_image(cropped_right)

            # eye identities
            participant_eyes_left.append((processed_left, labels[j]))
            participant_eyes_right.append((processed_right, labels[j]))

            # gaze identities
            gaze_pairs.append([(processed_left, labels[i]),
                               (processed_right, labels[i])])

        # each participant has 105 images so for each participant needs to have 52 left and 53 pairs or the other way around
        # the i % 2 makes sure that when i is even a extra left pair is added and when i is odd and extra right pair is added
        random_left = create_random_pair(participant_eyes_left, i % 2 == 0)
        random_right = create_random_pair(participant_eyes_right, i % 2 == 1)

        eye_pairs += random_left
        eye_pairs += random_right

    assert len(gaze_pairs) == len(eye_pairs)

    return gaze_pairs, eye_pairs


class ColumbiaGaze(Dataset):
    '''Dataset of (eye, gaze_label) entries for training a supervised model'''

    def __init__(self, path):
        gaze_pairs, _ = create_pair_dataset(path)
        self.eyes = [left for left, _ in gaze_pairs] + \
            [right for _, right in gaze_pairs]

    def __len__(self):
        return len(self.eyes)

    def __getitem__(self, idx):
        return self.eyes[idx]


class ColumbiaPairs(Dataset):
    '''Dataset of [(eye1, label), (eye2, label), (eye3, label), (eye4, label)] entries,
    (eye1, eye2) form a gaze-similar pair, (eye3, eye4) form an eye-similar pair'''

    def __init__(self, path, store_path='columbia_preprocessed.npy'):
        if os.path.isfile(store_path):
            print('loading preprocessed data')

            with open(store_path, 'rb') as f:
                self.data = torch.load(f)
            return

        gaze_pairs, eye_pairs = create_pair_dataset(path)

        self.data = [gaze_pair + eye_pair
                     for gaze_pair, eye_pair in zip(gaze_pairs, eye_pairs)]

        with open(store_path, 'wb') as f:
            torch.save(self.data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
