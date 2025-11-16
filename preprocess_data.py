import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split

from utils import Utility, Data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for training.")

    parser.add_argument("--input_dir", type=str, default="Directory where the data is stored")
    parser.add_argument("--output_dir", type=str, default="Directory where to put preprocessed data")
    args = parser.parse_args()

    data = Data(args.input_dir, True)
    data.load_train_data()
    X = data.kappa
    y = data.label

    # Split the data into training and validation sets
    NP_idx = np.arange(256)  # The indices of Nsys nuisance parameter realizations
    split_fraction = 0.2  # Set the fraction of data you want to split (between 0 and 1)
    seed = 5566  # Define your random seed for reproducible results

    train_NP_idx, val_NP_idx = train_test_split(NP_idx, test_size=split_fraction,
                                                random_state=seed)

    X_train = X[:, train_NP_idx]
    y_train = y[:, train_NP_idx, :2].reshape(-1, 2)
    X_val = X[:, val_NP_idx]
    y_val = y[:, val_NP_idx, :2].reshape(-1, 2)

    os.makedirs(args.output_dir, exist_ok=True)
    Utility.save_np(args.output_dir, "X_train.npy", X_train)
    Utility.save_np(args.output_dir, "y_train.npy", y_train)
    Utility.save_np(args.output_dir, "X_val.npy", X_val)
    Utility.save_np(args.output_dir, "y_val.npy", y_val)

    data.load_test_data()
    Utility.save_np(args.output_dir, "X_test.npy", data.kappa_test)