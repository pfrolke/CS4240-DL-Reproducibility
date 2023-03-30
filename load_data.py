import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.io import read_image
import torchvision.transforms.functional as fn


def crop_image(image, left_landmarks, right_landmarks):
    # Crop the image to the bounding box from the landmark of the left and right eye
    tlx, tly, brx, bry = left_landmarks
    cropped_left = fn.crop(image, tly, tlx, bry - tly, brx - tlx)
    padded_left = fn.center_crop(cropped_left, (153, 95))

    tlx, tly, brx, bry = right_landmarks
    cropped_right = fn.crop(image, tly, tlx, bry - tly, brx - tlx)
    padded_right = fn.center_crop(cropped_right, (153, 95))

    return padded_left, padded_right


def process_image(image):
    # Perform a histogram equalization and grayscaling on the tensor image
    equalized = fn.equalize(image)
    return equalized


def create_dataset_from_mix(path):
    data_set_left = []
    data_set_right = []

    # load labels and landmarks
    labels = np.load(os.path.join(path, 'labels.npy'))
    left_landmarks = np.load(os.path.join(path, 'left_landmark_mix.npy'))
    right_landmarks = np.load(os.path.join(path, 'right_landmark_mix.npy'))

    # load images from path
    print("Loading & preprocessing dataset 🚀")
    for i in tqdm(range(labels.shape[0])):
        img_path = os.path.join(path, f'{i}.png')
        img = read_image(img_path)

        # Crop the image
        cropped_left, cropped_right = crop_image(
            img, left_landmarks[i], right_landmarks[i])

        # Process the image
        processed_left = process_image(cropped_left)
        processed_right = process_image(cropped_right)

        # Append the processed image and label to their respective dataset
        data_set_left.append((processed_left, labels[i]))
        data_set_right.append((processed_right, labels[i]))

    return data_set_left, data_set_right

class ColumbiaGaze(Dataset):
    def __init__(self, path):
        data_set_left, data_set_right = create_dataset_from_mix(path)
        self.eyes = data_set_left + data_set_right

    def __len__(self):
        return len(self.eyes)

    def __getitem__(self, idx):
        return self.eyes[idx]
    
def create_eye_pair_dataset(path):
    data_set = []

    num_ids = len(os.listdir(path)) - 2 # 2 directories are eye_landmark and mix

    # For each participant crop and process the images and append that list to the dataset
    for i in range(1, num_ids+1):
        participant_path = os.path.join(path, str(i))

        participant_eyes = []

        labels = np.load(os.path.join(participant_path, 'labels.npy'))
        left_landmarks = np.load(os.path.join(participant_path, 'left_landmark_mix.npy'))
        right_landmarks = np.load(os.path.join(participant_path, 'right_landmark_mix.npy'))        

        num_imgs = len(os.listdir(participant_path))
        for j in range(num_imgs):
            img_path = os.path.join(path, f'{j}.png')
            img = read_image(img_path)

            # Crop the image
            cropped_left, cropped_right = crop_image(
                img, left_landmarks[j], right_landmarks[j])

            # Process the image
            processed_left = process_image(cropped_left)
            processed_right = process_image(cropped_right)

            participant_eyes.append((processed_left, labels[j]))
            participant_eyes.append((processed_right, labels[j]))
        
        data_set.append(participant_eyes)

    return data_set


