import json
import os
import zipfile

import numpy as np


class Utility:
    @staticmethod
    def add_noise(data, mask, ng, pixel_size=2.):
        """
        Add noise to a noiseless convergence map.

        Parameters
        ----------
        data : np.array
            Noiseless convergence maps.
        mask : np.array
            Binary mask map.
        ng : float
            Number of galaxies per arcmin². This determines the noise level.
        pixel_size : float, optional
            Pixel size in arcminutes (default is 2.0).
        """
        return data + np.random.randn(*data.shape) * 0.4 / (2 * ng * pixel_size ** 2) ** 0.5 * mask

    @staticmethod
    def load_np(data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        return np.load(file_path)

    @staticmethod
    def save_np(data_dir, file_name, data):
        file_path = os.path.join(data_dir, file_name)
        np.save(file_path, data)

    @staticmethod
    def save_json_zip(submission_dir, json_file_name, zip_file_name, data):
        """Save dictionary to JSON and compress into ZIP file."""
        os.makedirs(submission_dir, exist_ok=True)
        json_path = os.path.join(submission_dir, json_file_name)

        with open(json_path, "w") as f:
            json.dump(data, f)

        zip_path = os.path.join(submission_dir, zip_file_name)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname=json_file_name)

        os.remove(json_path)
        return zip_path


class Data:
    def __init__(self, data_dir, USE_PUBLIC_DATASET):
        self.USE_PUBLIC_DATASET = USE_PUBLIC_DATASET
        self.data_dir = data_dir
        self.mask_file = 'WIDE12H_bin2_2arcmin_mask.npy'
        self.viz_label_file = 'label.npy'
        if self.USE_PUBLIC_DATASET:
            self.kappa_file = 'WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = self.viz_label_file
            self.Ncosmo = 101  # Number of cosmologies in the entire training data
            self.Nsys = 256  # Number of systematic realizations in the entire training data
            self.test_kappa_file = 'WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 4000  # Number of instances in the test data
        else:
            self.kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = 'sampled_label.npy'
            self.Ncosmo = 3  # Number of cosmologies in the sampled training data
            self.Nsys = 30  # Number of systematic realizations in the sampled training data
            self.test_kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 3  # Number of instances in the sampled test data

        self.shape = [1424, 176]  # dimensions of each map
        self.pixelsize_arcmin = 2  # pixel size in arcmin
        self.pixelsize_radian = self.pixelsize_arcmin / 60 / 180 * np.pi  # pixel size in radian
        self.ng = 30  # galaxy number density. This determines the noise level of the experiment. Do not change this number.

    def load_train_data(self):
        self.mask = Utility.load_np(data_dir=self.data_dir,
                                    file_name=self.mask_file)  # A binary map that shows which parts of the sky are observed and which areas are blocked
        self.kappa = np.zeros((self.Ncosmo, self.Nsys, *self.shape), dtype=np.float16)
        self.kappa[:, :, self.mask] = Utility.load_np(data_dir=self.data_dir,
                                                      file_name=self.kappa_file)  # Training convergence maps
        self.label = Utility.load_np(data_dir=self.data_dir,
                                     file_name=self.label_file)  # Training labels (cosmological and physical paramameters) of each training map
        self.viz_label = Utility.load_np(data_dir=self.data_dir,
                                         file_name=self.viz_label_file)  # For visualization of parameter distributions

    def load_train_labels(self):
        self.label = Utility.load_np(data_dir=self.data_dir,
                                     file_name=self.label_file)  # Training labels (cosmological and physical paramameters) of each training map
        self.viz_label = Utility.load_np(data_dir=self.data_dir,
                                         file_name=self.viz_label_file)  # For visualization of parameter distributions

    def load_test_data(self):
        if not hasattr(self, "mask"):
            self.mask = Utility.load_np(data_dir=self.data_dir,
                                        file_name=self.mask_file)  # A binary map that shows which parts of the sky are observed and which areas are blocked

        self.kappa_test = np.zeros((self.Ntest, *self.shape), dtype=np.float16)
        self.kappa_test[:, self.mask] = Utility.load_np(data_dir=self.data_dir,
                                                        file_name=self.test_kappa_file)  # Test noisy convergence maps


class Score:
    @staticmethod
    def _score_phase1(true_cosmo, infer_cosmo, errorbar):
        """Compute log-likelihood score for Phase 1."""
        sq_error = (true_cosmo - infer_cosmo) ** 2
        scale_factor = 1000
        score = - np.sum(sq_error / errorbar ** 2 + np.log(errorbar ** 2) + scale_factor * sq_error, 1)
        score = np.mean(score)
        print("MSE", np.mean(sq_error))
        if score >= -10 ** 6:
            return score
        else:
            return -10 ** 6
