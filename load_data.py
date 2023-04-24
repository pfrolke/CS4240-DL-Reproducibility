import numpy as np
import os
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
from torchvision.io import read_image
import torchvision.transforms.functional as fn
import torchvision.transforms as transforms

NUM_SUBJECTS = 56
NUM_SUBJECT_IMAGES = 105


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
    grey = fn.rgb_to_grayscale(equalized)
    resized = transforms.Resize((32, 64))(grey)
    return resized


def find_max_landmark(path):
    '''Find maximum landmark width and height'''
    max_width = -1
    max_height = -1

    for i in range(1, NUM_SUBJECTS + 1):
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
    all_labels = []

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

        all_labels.extend(labels)

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

            # eye similar pair
            participant_eyes_left.append(processed_left)
            participant_eyes_right.append(processed_right)

            # gaze similar pair
            gaze_pair = [processed_left, processed_right]
            random.shuffle(gaze_pair)
            gaze_pairs.append(gaze_pair)

        # each participant has 105 images so for each participant needs to have 52 left and 53 pairs or the other way around
        # the i % 2 makes sure that when i is even a extra left pair is added and when i is odd and extra right pair is added
        random_left = create_random_pair(participant_eyes_left, i % 2)
        random_right = create_random_pair(participant_eyes_right, i % 2)

        eye_pairs += random_left
        eye_pairs += random_right

    assert len(gaze_pairs) == len(eye_pairs)

    return gaze_pairs, eye_pairs, all_labels


class ColumbiaGaze(Dataset):
    '''Dataset of (eye, gaze_label) entries for training a supervised model'''

    def __init__(self, path):
        gaze_pairs, _, labels = create_pair_dataset(path)
        self.eyes = []

        for (eye1, eye2), label in zip(gaze_pairs, labels):
            self.eyes.append((eye1, label))
            self.eyes.append((eye2, label))

    def __len__(self):
        return len(self.eyes)

    def __getitem__(self, idx):
        return self.eyes[idx]


class ColumbiaPairs(Dataset):
    '''Dataset of [eye1, eye2, eye3, eye4] entries,
    (eye1, eye2) form a gaze-similar pair, (eye3, eye4) form an eye-similar pair
    __getitem__ also returns the gaze labels for eye1 and eye2'''

    def __init__(self, data_path, subjects, store_path='columbia_preprocessed.pt'):
        if os.path.isfile(store_path):
            print('loading preprocessed data')

            with open(store_path, 'rb') as f:
                loaded = torch.load(f)

            data = loaded['data']
            self.labels = loaded['labels']
        else:
            gaze_pairs, eye_pairs, self.labels = create_pair_dataset(data_path)

            # concatenate pairs and convert to tensor
            data = torch.stack([torch.cat((torch.stack(gaze_pair), torch.stack(eye_pair)))
                                for gaze_pair, eye_pair in zip(gaze_pairs, eye_pairs)])

            with open(store_path, 'wb') as f:
                torch.save({'data': data, 'labels': self.labels}, f)

        # select only images for specified subject
        select_indices = torch.flatten(torch.tensor([
            range(NUM_SUBJECT_IMAGES * i, NUM_SUBJECT_IMAGES * (i + 1)) for i in subjects]))
        self.selected_data = torch.index_select(
            data, dim=0, index=select_indices)

    def __len__(self):
        return len(self.selected_data)

    def __getitem__(self, idx):
        return self.selected_data[idx], self.labels[idx]
