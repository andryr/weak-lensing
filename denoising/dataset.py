import torch
from torch.utils.data import Dataset

from utils import Utility


class DenoisingDataset(Dataset):
    """Dataset for full weak lensing convergence maps (no patches)"""

    def __init__(self, kappa_clean, mask, ng, pixel_size, labels=None, transform=None, ):
        """
        Args:
            kappa_clean: Clean convergence maps (B, H, W)
            mask: Binary mask for the observation region
            ng: Galaxy number density (noise level)
            pixel_size: Pixel size in arcminutes
            labels: Optional labels for cosmological parameters (B, n_params)
        """
        self.kappa_clean = kappa_clean
        self.mask = mask
        self.ng = ng
        self.pixel_size = pixel_size
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.kappa_clean)

    def __getitem__(self, idx):
        # Get clean map
        clean = self.kappa_clean[idx]

        # Add noise on-the-fly
        noisy = Utility.add_noise(clean, self.mask, self.ng, self.pixel_size)
        if self.transform is not None:
            noisy = self.transform(noisy)
        # noisy = _apply_coarse_dropout(noisy)

        # Convert to tensors and add channel dimension: (H, W) -> (1, H, W)
        mean = -0.00014706686488352716
        std = 0.008249041624367237
        clean = (torch.FloatTensor(clean).unsqueeze(0) - mean) / std
        noisy = (torch.FloatTensor(noisy).unsqueeze(0) - mean) / std

        if self.labels is not None:
            label = torch.FloatTensor(self.labels[idx])
            return noisy, clean, label

        return noisy, clean