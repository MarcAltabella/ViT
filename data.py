"""
Olivetti faces, cached locally by scikit-learn.

40 people, 10 photos each. Each image is 64x64 grayscale, values in [0, 1].
A batch is an image tensor [B, 1, 64, 64] and class indices [B].

Match the model to this data:
    img_size=64, n_channels=1, patch_size=8, n_classes=40
patch_size must divide img_size, and d_model must divide evenly by n_heads.
"""

from pathlib import Path

import torch
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, TensorDataset

DATA_HOME = Path(__file__).resolve().parent / "data" / "sklearn"

IMG_SIZE = 64
N_CHANNELS = 1
N_CLASSES = 40


def get_dataloader(
    batch_size: int = 32, shuffle: bool = True, train: bool = True
) -> DataLoader:
    pixels, targets = fetch_olivetti_faces(
        data_home=DATA_HOME,
        shuffle=False,
        download_if_missing=False,
        return_X_y=True,
    )

    images = torch.from_numpy(pixels).reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    labels = torch.from_numpy(targets).long()

    # 10 photos per person: 8 for training, 2 for eval
    keep = []
    for start in range(0, len(labels), 10):
        if train:
            keep.extend(range(start, start + 8))
        else:
            keep.extend(range(start + 8, start + 10))

    dataset = TensorDataset(images[keep], labels[keep])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
