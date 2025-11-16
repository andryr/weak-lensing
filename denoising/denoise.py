import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from denoising.dataset import DenoisingDataset
from denoising.model import DenoisingUNet
from utils import Utility


def denoise(args):
    model = DenoisingUNet.load_from_checkpoint(
        args.model
    )
    X = Utility.load_np(data_dir=".", file_name=args.input)
    mask = Utility.load_np(data_dir=".", file_name=args.mask)

    pixelsize_arcmin = 2  # pixel size in arcmin
    ng = 30
    dataset = DenoisingDataset(
        X,
        mask,
        ng,
        pixelsize_arcmin
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )

    model.eval()
    denoised = []
    for batch in tqdm(loader, desc="Denoising"):
        if args.add_noise:
            x_noisy = batch[0].cuda()
        else:
            x_noisy = batch[1].cuda()
        with torch.no_grad():
            _, x_denoised = model(x_noisy)
            x_denoised = x_denoised.cpu().numpy().squeeze(1)
            # Denormalize back to original scale
            mean = -0.00014706686488352716
            std = 0.008249041624367237
            x_denoised = x_denoised * std + mean
        denoised.append(x_denoised)
    denoised = np.vstack(denoised)
    Utility.save_np(data_dir=".", file_name=args.output, data=denoised)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise convergence maps.")
    parser.add_argument("--model", type=str,
                        help="Path to the denoising model")
    parser.add_argument("--input", type=str,
                        help="Path to input numpy file")
    parser.add_argument("--add_noise", action=argparse.BooleanOptionalAction,
                        help="Whether to add noise before denoising")
    parser.add_argument("--mask", type=str, default="WIDE12H_bin2_2arcmin_mask.npy",
                        help="Path to the mask file (default: WIDE12H_bin2_2arcmin_mask.npy)")
    parser.add_argument("--output", type=str,
                        help="Path to output denoised numpy file")
    args = parser.parse_args()
    denoise(args)
