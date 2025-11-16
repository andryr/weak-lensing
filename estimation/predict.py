import argparse
import datetime
import time

import numpy as np
import skops.io as sio
import timm
import torch
from scipy.interpolate import RBFInterpolator, NearestNDInterpolator
from scipy.stats import gamma, beta
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from estimation.dataset import transform_test, CosmologyDataset
from estimation.model import EnsembleModel, ParameterEstimator
from utils import Utility, Score


def init_single_model(cnn_model_name, cnn_feat_dim):
    # Initialize the CNN model
    cnn = timm.create_model(cnn_model_name, pretrained=True, num_classes=0)
    model = ParameterEstimator(
        cnn,
        cnn_feat_dim,
        2,
    )
    return model


def predict(args):
    cnn_models = [
        init_single_model(model_name, feat_dim) for model_name, feat_dim in
        [
            ('regnetz_040.ra3_in1k', 528),
            ('efficientnet_b3', 1536),
            ('efficientnet_b2a', 1408),
            ('regnetv_040.ra3_in1k', 1088),
            ('regnety_040.ra3_in1k', 1088),
        ]
    ]
    model = EnsembleModel(cnn_models)
    X_val = Utility.load_np(data_dir=".", file_name=args.x_val)
    y_val = Utility.load_np(data_dir=".", file_name=args.y_val)
    val_dataset = CosmologyDataset(
        data=X_val,
        transform=transform_test
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )
    X_test = Utility.load_np(data_dir=".", file_name=args.x_test)
    test_dataset = CosmologyDataset(
        data=X_test,
        transform=transform_test
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )

    label_scaler = sio.load("label_scaler.skops")
    pca = sio.load("label_pca.skops")

    model.eval()
    y_pred_val = point_prediction(model, val_loader, label_scaler, pca)
    y_pred_test = point_prediction(model, test_loader, label_scaler, pca)

    print(f"Validation MSE: {np.mean((y_val - y_pred_val) ** 2, axis=0)}")

    # There are Ncosmo distinct cosmologies in the labels.
    # Here we create a list that groups the indices of the validation instances with the same cosmological parameters
    cosmology = Utility.load_np(data_dir=".", file_name=args.full_label_array)
    cosmology = cosmology.label[:, 0, :2]  # shape = (Ncosmo, 2)

    Ncosmo = 101
    row_to_i = {tuple(cosmology[i]): i for i in range(Ncosmo)}
    index_lists = [[] for _ in range(cosmology.shape[0])]

    # Loop over each row in 'y_val' with shape = (Nval, 2)
    for idx in range(len(y_val)):
        row_tuple = tuple(y_val[idx])
        i = row_to_i[row_tuple]
        index_lists[i].append(idx)

    # val_cosmology_idx[i] = the indices idx of the validation examples with labels = cosmology[i]
    val_cosmology_idx = [np.array(lst) for lst in index_lists]

    # The summary statistics of all realizations for all cosmologies in the validation set
    d_vector = []
    n_d = 2  # Number of summary statistics for each map
    for i in range(Ncosmo):
        d_i = np.zeros((len(val_cosmology_idx[i]), n_d))
        for j, idx in enumerate(val_cosmology_idx[i]):
            d_i[j] = y_pred_val[idx]

        d_vector.append(d_i)

    # mean summary statistics (average over all realizations)
    mean_d_vector = []
    for i in range(Ncosmo):
        mean_d_vector.append(np.mean(d_vector[i], 0))
    mean_d_vector = np.array(mean_d_vector)

    # covariance matrix
    delta = []
    for i in range(Ncosmo):
        delta.append((d_vector[i] - mean_d_vector[i].reshape(1, n_d)))

    cov_d_vector = [(delta[i].T @ delta[i] / (len(delta[i]) - n_d - 2))[None] for i in range(Ncosmo)]
    cov_d_vector = np.concatenate(cov_d_vector, 0)

    mean_d_vector_interp = RBFInterpolator(cosmology, mean_d_vector, kernel='thin_plate_spline', neighbors=10)
    cov_d_vector_interp = NearestNDInterpolator(cosmology, cov_d_vector)

    sio.dump(mean_d_vector_interp, "mean_d_vector_interp.skops")
    sio.dump(cov_d_vector_interp, "cov_d_vector_interp.skops")

    om = cosmology[:, 0]
    om_gamma_shape, om_gamma_loc, om_gamma_scale = gamma.fit(om)
    s8 = cosmology[:, 1]
    s8_beta_a, s8_beta_b, s8_beta_loc, s8_beta_scale = beta.fit(s8)

    def log_prior(x):
        return gamma.logpdf(x[:, 0], om_gamma_shape, om_gamma_loc, om_gamma_scale) + beta.logpdf(x[:, 1], a=s8_beta_a,
                                                                                                 b=s8_beta_b,
                                                                                                 loc=s8_beta_loc,
                                                                                                 scale=s8_beta_scale)

    # Gaussian likelihood with interpolated mean and covariance matrix
    def loglike(x, d):
        mean = mean_d_vector_interp(x)
        cov = cov_d_vector_interp(x)
        delta = d - mean

        inv_cov = np.linalg.inv(cov)
        cov_det = np.linalg.slogdet(cov)[1]

        return -0.5 * cov_det - 0.5 * np.einsum("ni,nij,nj->n", delta, inv_cov, delta)

    def logp_posterior(x, d):
        logp = log_prior(x)
        select = np.isfinite(logp)
        if np.sum(select) > 0:
            logp[select] = logp[select] + loglike(x[select], d[select])
        return logp

    def mcmc(y_pred, num_steps, sigma, num_chains):
        # Randomly select initial points from the `cosmology` array for each test case
        # Assumes `cosmology` has shape (Ncosmo, ndim) and `Ntest` is the number of independent chains/samples
        current = cosmology[np.random.choice(Ncosmo, size=num_chains)]

        # Compute log-posterior at the initial points
        curr_logprob = logp_posterior(current, y_pred)

        # List to store sampled states (for all chains)
        states = []

        # Track total acceptance probabilities to compute acceptance rates
        total_acc = np.zeros(len(current))

        t = time.time()  # Track time for performance reporting

        # MCMC loop
        for i in trange(num_steps):

            # Generate proposals by adding Gaussian noise to current state
            proposal = current + np.random.randn(*current.shape) * sigma

            # Compute log-posterior at the proposed points
            proposal_logprob = logp_posterior(proposal, y_pred)

            # Compute log acceptance ratio (Metropolis-Hastings)
            acc_logprob = proposal_logprob - curr_logprob
            acc_logprob[acc_logprob > 0] = 0  # Cap at 0 to avoid exp overflow (acceptance prob ≤ 1)

            # Convert to acceptance probabilities
            acc_prob = np.exp(acc_logprob)

            # Decide whether to accept each proposal
            acc = np.random.uniform(size=len(current)) < acc_prob

            # Track acceptance probabilities (not binary outcomes)
            total_acc += acc_prob

            # Update states and log-probs where proposals are accepted
            current[acc] = proposal[acc]
            curr_logprob[acc] = proposal_logprob[acc]

            # Save a copy of the current state
            states.append(np.copy(current)[None])

            # Periodically print progress and acceptance rates
            if i % (0.1 * num_steps) == 0.1 * num_steps - 1:
                print(
                    'step:', len(states),
                    'Time:', time.time() - t,
                    'Min acceptance rate:', np.min(total_acc / (i + 1)),
                    'Mean acceptance rate:', np.mean(total_acc / (i + 1))
                )
                t = time.time()  # Reset timer for next print interval
        # remove burn-in
        states = np.concatenate(states[int(0.2 * len(states)):], 0)

        # mean and std of samples
        mean = np.mean(states, 0)
        errorbar = np.std(states, 0)
        return mean, errorbar

    mean_val, errorbar_val = mcmc(y_pred_val, num_steps=1000, sigma=0.06, num_chains=y_pred_val.shape[0])
    validation_score = Score._score_phase1(
        true_cosmo=y_val,
        infer_cosmo=mean_val,
        errorbar=errorbar_val
    )
    print('averaged score:', np.mean(validation_score))
    print('averaged error bar:', np.mean(errorbar_val, 0))

    mean, errorbar = mcmc(y_pred_test, num_steps=10000, sigma=0.06, num_chains=y_pred_test.shape[0])
    data = {"means": mean.tolist(), "errorbars": errorbar.tolist()}
    the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    zip_file_name = 'Submission_' + the_date + '.zip'
    zip_file = Utility.save_json_zip(
        submission_dir="submissions",
        json_file_name="result.json",
        zip_file_name=zip_file_name,
        data=data
    )
    print(f"Submission ZIP saved at: {zip_file}")


def point_prediction(model, loader, label_scaler, pca):
    model.eval()
    y_pred_list = []
    pbar = tqdm(loader, total=len(loader), desc="Inference on the test set")
    with torch.no_grad():
        for X in pbar:
            batch_size = X.shape[0]
            X = X.cuda()
            y_pred = model(X)
            y_pred = pca.inverse_transform(label_scaler.inverse_transform(y_pred.cpu().numpy())).reshape(batch_size, -1)
            y_pred_list.append(y_pred)
    return np.concatenate(y_pred_list, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise convergence maps.")
    parser.add_argument("--model", type=str,
                        help="Path to the denoising model")
    parser.add_argument("--x_val", type=str, default="X_val.npy", help="Path to X_val numpy file (default: X_val.npy)")
    parser.add_argument("--y_val", type=str, default="y_val.npy", help="Path to y_val numpy file (default: y_val.npy)")
    parser.add_argument("--x_test", type=str, default="X_test.npy", help="Path to X_val numpy file (default: X_test.npy)")

    args = parser.parse_args()
    predict(args)
