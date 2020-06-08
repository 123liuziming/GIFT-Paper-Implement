import os
import pickle

import numpy as np
import torch


def normalize_image(img, mask=None):
    if mask is not None: img[np.logical_not(mask.astype(np.bool))] = 127
    img = (img.transpose([2, 0, 1]).astype(np.float32) - 127.0) / 128.0
    return torch.tensor(img, dtype=torch.float32)


def gray_repeats(img_raw):
    if len(img_raw.shape) == 2:
        img_raw = np.repeat(img_raw[:, :, None], 3, axis=2)
    if img_raw.shape[2] > 3:
        img_raw = img_raw[:, :, :3]
    return img_raw


def perspective_transform(pts, H):
    tpts = np.concatenate([pts, np.ones([pts.shape[0], 1])], 1) @ H.transpose()
    tpts = tpts[:, :2] / np.abs(tpts[:, 2:])  # todo: why only abs? this one is correct
    return tpts


def get_rot_m(angle):
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)  # rn+1,3,3


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
