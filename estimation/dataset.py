import numpy as np
import torch
from torch.utils.data import Dataset


class CosmologyDataset(Dataset):
    """
    Custom PyTorch Dataset
    """

    def __init__(self, data, labels=None,
                 transform=None,
                 label_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float32)  # [np.newaxis, ...]
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx].astype(np.float32)
            label = torch.from_numpy(label)
            if self.label_transform:
                label = self.label_transform(label)
            return image, label
        else:
            return image


from torchvision.transforms import v2
import random
import albumentations as A

# use albumentations properly by instantiating and wrapping it in a callable
_coarse_dropout = A.CoarseDropout(
    num_holes_range=(12, 24),
    hole_height_range=(8, 24),
    hole_width_range=(8, 24),
    fill="random_uniform",
    p=1.0,
)


def _apply_coarse_dropout(img):
    # albumentations expects HWC numpy arrays. If input is 2D (H, W), add channel dim,
    # apply transform and remove the added channel afterwards.
    added_channel = False
    if img.ndim == 2:
        img = img[..., None]  # (H, W) -> (H, W, 1)
        added_channel = True
    out = _coarse_dropout(image=img)["image"]
    if added_channel:
        out = out[..., 0]  # (H, W, 1) -> (H, W)
    return out





def expand(x):
    return x.expand(3, -1, -1)


def extract_patches(x):
    # x: (C, H, W), expects (3, 1424, 176)
    patches = []
    # Extract 88x88 patches in the first column
    for row_start in range(0, x.shape[1] - 88 + 1, 88):
        patch = x[:, row_start:row_start + 88, :88]
        patches.append(patch)
    # Extract one additional 88x88 patch at (620, 88)
    patch_extra = x[:, 620:708, 88:176]
    patches.append(patch_extra)
    patches = torch.stack(patches, dim=0)  # shape: (num_patches, 3, 88, 88)
    return patches


def extract_random_patches(x, n=16):
    patches = []
    # Randomly select 16 starting rows for the first column patches
    max_row_start = x.shape[1] - 88
    row_starts = random.sample(range(0, max_row_start + 1), n)
    for row_start in row_starts:
        patch = x[:, row_start:row_start + 88, :88]
        patches.append(patch)
    # Extract one additional 88x88 patch at a random row between 620 and 700
    extra_row_start = random.randint(620, 700)
    patch_extra = x[:, extra_row_start:extra_row_start + 88, 88:176]
    patches.append(patch_extra)
    patches = torch.stack(patches, dim=0)  # shape: (17, 3, 88, 88)
    return patches


pixelsize_arcmin = 2  # pixel size in arcmin
pixelsize_radian = pixelsize_arcmin / 60 / 180 * np.pi  # pixel size in radian
ng = 30

means = -0.00014706686488352716
stds = 0.008249041624367237
sigma = 0.4 / (2 * ng * pixelsize_arcmin ** 2) ** 0.5
transform = v2.Compose(
    [
        # partial(denoise_bilateral, sigma_color=sigma),
        torch.from_numpy,
        lambda x: x.unsqueeze(0),  # add channel dimension
        expand,
        torch.Tensor.float,
        # v2.Resize(224),
        v2.Normalize(mean=[float(means)], std=[float(stds)]),
    ]
)
transform_val = v2.Compose(
    [
        transform,
        extract_patches,
    ]
)
transform_test = v2.Compose(
    [
        transform,
        extract_patches
    ]
)
transform_with_aug = v2.Compose(
    [
        _apply_coarse_dropout,
        transform,
        extract_random_patches,
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        lambda x: x.transpose(-2, -1) if np.random.rand() > 0.5 else x,
    ]
)
