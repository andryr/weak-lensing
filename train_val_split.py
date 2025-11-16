import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from utils import Utility

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/validation data split.")

    parser.add_argument("--images", type=str, default="WIDE12H_bin2_2arcmin_kappa.npy",
                        help="Path to numpy file containing images (default: WIDE12H_bin2_2arcmin_kappa.npy)")
    parser.add_argument("--labels", type=str, default="labels.npy", help="Path numpy file containing labels (default: labels.npy)")
    args = parser.parse_args()

    X = Utility.load_np(".", args.images)
    y = Utility.load_np(".", args.labels)
    # Split the data into training and validation sets

    NP_idx = np.arange(256)  # The indices of Nsys nuisance parameter realizations
    split_fraction = 0.2  # Set the fraction of data you want to split (between 0 and 1)
    seed = 5566  # Define your random seed for reproducible results

    train_NP_idx, val_NP_idx = train_test_split(NP_idx, test_size=split_fraction,
                                                random_state=seed)

    X_train = X[:, train_NP_idx]
    y_train = y[:, train_NP_idx]
    X_val = X[:, val_NP_idx]
    y_val = y[:, val_NP_idx]

    Utility.save_np(".", "X_train.npy", X_train)
    Utility.save_np(".", "X_val.npy", X_val)