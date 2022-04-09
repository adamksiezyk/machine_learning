import os
from typing import Tuple

import numpy as np

from ycb_video import CONFIG


def load() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Load keypoints
    kps = np.load(CONFIG['keypoints_path'])
    # Dataset size
    n = kps.shape[0]
    # Load all images
    file_names = os.listdir(CONFIG['images_path'])
    # Extract n images
    imgs = np.array(file_names[:2 * n + 1:2])
    # Extract n labels
    labels = np.array(file_names[1:2 * n + 1:2])
    return imgs, kps, labels


def batch_generator(x: np.ndarray, y: np.ndarray, batch_size: int):
    while True:
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            yield x[start:end], y[start:end]
