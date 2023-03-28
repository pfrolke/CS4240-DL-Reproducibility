import numpy as np
import os


def create_dataset(path, num_ids):
    images = {}

    for id in range(1, num_ids):
        path_id = os.path.join(path, str(id))
        labels = np.load(os.path.join(path_id, 'labels.npy'))
        left_landmarks = np.load(os.path.join(path_id, 'left_landmarks.npy'))
        right_landmarks = np.load(os.path.join(path_id, 'right_landmarks.npy'))

        eyes
