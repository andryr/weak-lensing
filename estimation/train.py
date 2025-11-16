import argparse
import time

import numpy as np
import pytorch_lightning as pl
import skops.io as sio
import timm
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.interpolate import RBFInterpolator, NearestNDInterpolator
from scipy.stats import gamma, beta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.optim.swa_utils import get_ema_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from estimation.dataset import CosmologyDataset, transform_with_aug, transform_val
from estimation.model import ParameterEstimator, EnsembleModel
from estimation.weight_averaging import WeightAveraging
from utils import Utility


class EMAWeightAveraging(WeightAveraging):
    def __init__(self):
        super().__init__(avg_fn=get_ema_avg_fn(decay=0.95))

    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= 100)


class LossHistoryCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())


def train_single_model(train_loader, val_loader, cnn_model_name, cnn_feat_dim, learning_rate, epochs):
    # Initialize the CNN model
    cnn = timm.create_model(cnn_model_name, pretrained=True, num_classes=0)
    model = ParameterEstimator(
        cnn,
        cnn_feat_dim,
        2,
        learning_rate=learning_rate,
        weight_decay=1e-2,
    )

    ema_callback = EMAWeightAveraging()

    loss_history_callback = LossHistoryCallback()
    # Set up Lightning callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='estimator_checkpoints/',
        filename='estimator-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        precision="16-mixed",
        devices=1,
        callbacks=[checkpoint_callback, loss_history_callback, ema_callback],
        log_every_n_steps=50,
        enable_progress_bar=True
    )

    # Train the model
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    end_time = time.time()

    print(f"\\nTraining finished in {(end_time - start_time) / 60:.2f} minutes.")

    plt.plot(loss_history_callback.train_losses[1:], label="Train Loss")
    plt.plot(loss_history_callback.val_losses[2:], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return model


def train(args):
    # Load data
    X_train = Utility.load_np(data_dir=".", file_name=args.x_train)
    X_val = Utility.load_np(data_dir=".", file_name=args.x_val)
    y_train = Utility.load_np(data_dir=".", file_name=args.y_train)
    y_val = Utility.load_np(data_dir=".", file_name=args.y_val)

    # Label standardization
    label_scaler = StandardScaler()
    pca = PCA()
    y_train_pca = pca.fit_transform(y_train)
    y_train_scaled = label_scaler.fit_transform(y_train_pca)
    y_val_scaled = label_scaler.transform(pca.transform(y_val))

    sio.dump(label_scaler, "label_scaler.skops")
    sio.dump(pca, "label_pca.skops")

    print(
        f"Label stats (from train set): Mean={label_scaler.mean_}, Std={np.sqrt(label_scaler.var_)}"
    )

    train_dataset = CosmologyDataset(
        data=X_train,
        labels=y_train_scaled,
        transform=transform_with_aug
    )
    val_dataset = CosmologyDataset(
        data=X_val,
        labels=y_val_scaled,
        transform=transform_val
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=3, shuffle=False)

    cnn_models = [
        ('regnetz_040.ra3_in1k', 528),
        ('efficientnet_b3', 1536),
        ('efficientnet_b2a', 1408),
        ('regnetv_040.ra3_in1k', 1088),
        ('regnety_040.ra3_in1k', 1088),
    ]
    ensemble_models = []
    for i, (cnn_model_name, cnn_model_feat_dim) in enumerate(tqdm(cnn_models)):
        print(f"Training model {i + 1}/{len(cnn_models)}...")
        model = train_single_model(train_loader, val_loader, cnn_model_name, cnn_model_feat_dim, args.lr, args.epochs)
        ensemble_models.append(model)

    model = EnsembleModel(ensemble_models)
    torch.save(model.state_dict(), "ensemble_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a denoising UNet.")
    parser.add_argument("--x_train", type=str, default="X_train.npy",
                        help="Path to X_train numpy file (default: X_train.npy)")
    parser.add_argument("--x_val", type=str, default="X_val.npy", help="Path to X_val numpy file (default: X_val.npy)")
    parser.add_argument("--y_train", type=str, default="y_train.npy",
                        help="Path to X_train numpy file (default: X_train.npy)")
    parser.add_argument("--y_val", type=str, default="y_val.npy", help="Path to y_val numpy file (default: y_val.npy)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    train(args)
