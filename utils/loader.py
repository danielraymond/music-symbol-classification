import os

import numpy as np
import torch

import config.data_config as config
from utils.parser import parse_files
from utils.patches import load_patches
from utils.preprocessing import filter_by_occurrence, get_w2i_dictionary


def load_pretrain_data(
    ds_name: str,
    num_random_patches: int = -1,
    kernel: tuple = (64, 64),
    stride: tuple = (32, 32),
    entropy_threshold: float = 0.8,
) -> torch.Tensor:
    config.set_data_dirs(ds_name=ds_name)
    os.makedirs(config.patches_dir, exist_ok=True)
    patches_filepath = "patches_"
    patches_filepath += f"k{'x'.join(map(str, kernel))}_"
    patches_filepath += f"s{'x'.join(map(str, stride))}_"
    patches_filepath += f"et{entropy_threshold}.npy"
    patches_filepath = config.patches_dir / patches_filepath

    print(patches_filepath)

    X = load_patches(
        patches_filepath=str(patches_filepath),
        kernel=kernel,
        stride=stride,
        entropy_threshold=entropy_threshold,
    )
    if num_random_patches > 0:
        perm = torch.randperm(len(X))
        X = X[perm[:num_random_patches]]

    return X


def load_supervised_data(ds_name: str, min_occurence: int = 50) -> dict:
    # 1) Parse files
    X, Y = parse_files(ds_name=ds_name)

    # 2) Filter out samples with low occurrence
    X, Y = filter_by_occurrence(bboxes=X, labels=Y, min_occurence=min_occurence)

    # 3) Get w2i dictionary
    w2i = get_w2i_dictionary(labels=Y)

    # 4) Preprocessing
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray([w2i[w] for w in Y], dtype=np.int64)

    return {"X": X, "Y": Y, "w2i": w2i}
