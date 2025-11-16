import argparse

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from denoising.dataset import DenoisingDataset
from denoising.model import DenoisingUNet
from utils import Utility


def train(args):
    # Load data
    X_train = Utility.load_np(data_dir=".", file_name=args.x_train)
    X_val = Utility.load_np(data_dir=".", file_name=args.x_val)
    mask = Utility.load_np(data_dir=".", file_name=args.mask)

    pixelsize_arcmin = 2  # pixel size in arcmin
    pixelsize_radian = pixelsize_arcmin / 60 / 180 * np.pi  # pixel size in radian
    ng = 30
    train_dataset = DenoisingDataset(
        X_train,
        mask,
        ng,
        pixelsize_arcmin
    )

    val_dataset = DenoisingDataset(
        X_val,
        mask,
        ng,
        pixelsize_arcmin,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2 * args.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='denoiser_checkpoints/',
        filename='autoencoder-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        precision="16-mixed",
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )
    model = DenoisingUNet(
        feature_dim=512,
        learning_rate=args.lr,
        weight_decay=1e-4
    )
    print("Trainer initialized. Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a denoising UNet.")
    parser.add_argument("--mask", type=str, default="WIDE12H_bin2_2arcmin_mask.npy",
                        help="Path to the mask file (default: WIDE12H_bin2_2arcmin_mask.npy)")
    parser.add_argument("--x_train", type=str, default="X_train.npy",
                        help="Path to X_train numpy file (default: X_train.npy)")
    parser.add_argument("--x_val", type=str, default="X_val.npy", help="Path to X_val numpy file (default: X_val.npy)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    train(args)
