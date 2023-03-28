import numpy as np
import os
from torchvision.io import read_image
import torchvision.transforms.functional as fn
import torchvision.transforms as transforms


def crop_image(image, left_landmarks, right_landmarks):
    # Crop the image to the bounding box from the landmark of the left and right eye
    tlx, tly, brx, bry = left_landmarks
    cropped_left = fn.crop(image, tly, tlx, bry - tly, brx - tlx)

    tlx, tly, brx, bry = right_landmarks
    cropped_right = fn.crop(image, tly, tlx, bry - tly, brx - tlx)
    return cropped_left, cropped_right

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
    num_ids = len(os.listdir(path)) - 3 # 3 files are labels, left_landmarks, right_landmarks


    # load images from path
    for i in range(num_ids):
        img_path = os.path.join(path, str(i) + '.png')
        img = read_image(img_path)

        # Crop the image 
        cropped_left, cropped_right = crop_image(img, left_landmarks[i], right_landmarks[i])

        # Process the image
        processed_left = process_image(cropped_left)
        processed_right = process_image(cropped_right)

        # Append the processed image and label to their respective dataset
        data_set_left.append((processed_left, labels[i]))
        data_set_right.append((processed_right, labels[i]))

    return data_set_left, data_set_right

