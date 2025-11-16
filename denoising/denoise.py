import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from denoising.model import DenoisingUNet
from utils import Utility


def denoise(args):
    model = DenoisingUNet.load_from_checkpoint(
        args.model
    )
    X = Utility.load_np(data_dir=".", file_name=args.noisy)
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )

    model.eval()
    denoised = []
    for batch in loader:
        x_noisy = batch[0]
        with torch.no_grad():
            x_denoised = model(x_noisy)
        denoised.append(x_denoised)
    denoised = torch.cat(denoised, dim=0).cpu().numpy()
    Utility.save_np(data_dir=".", file_name=args.denoised, data=denoised)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise convergence maps.")
    parser.add_argument("--model", type=str,
                        help="Path to the denoising model")
    parser.add_argument("--noisy", type=str,
                        help="Path to input numpy file")
    parser.add_argument("--denoised", type=str,
                        help="Path to output denoised numpy file")
    args = parser.parse_args()
    denoise(args)
