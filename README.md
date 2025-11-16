# Weak Lensing Submission

This is my submission for the Neurips 2025 Weak Lensing Challenge. The project consists of two main components: a denoising U-Net model and a cosmological parameter estimator using an ensemble of CNN architectures.
## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install dependencies with:

```bash
uv sync
```

## Workflow

The typical workflow consists of the following steps:

### 1. Prepare Data

Run the data preparation script to split the dataset into training and validation sets, and prepare the test data.
```bash
uv run -m preprocess_data --input_dir dataset_dir --output_dir preprocess_data_dir
```

**Arguments:**
- `--images`: Path to numpy file containing images (default: `WIDE12H_bin2_2arcmin_kappa.npy`)
- `--labels`: Path to numpy file containing labels (default: `labels.npy`)

**Output:**
- `X_train.npy`: Training images
- `X_val.npy`: Validation images

### 2. Train the Denoiser

Train a denoising U-Net model on the training data:

```bash
uv run -m denoising.train \
  --mask WIDE12H_bin2_2arcmin_mask.npy \
  --x_train X_train.npy \
  --x_val X_val.npy \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4
```

**Arguments:**
- `--mask`: Path to the mask file (default: `WIDE12H_bin2_2arcmin_mask.npy`)
- `--x_train`: Path to training images (default: `X_train.npy`)
- `--x_val`: Path to validation images (default: `X_val.npy`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)

**Output:**
- Model checkpoints saved in `denoiser_checkpoints/`

### 3. Generate Denoised Data

Apply the trained denoiser to generate denoised versions of the train/val data. For training data, noise is added before denoising to simulate the noisy conditions.

```bash
# Denoise training data
uv run -m denoising.denoise \
  --model denoiser_checkpoints/best_model.ckpt \
  --input X_train.npy \
  --add_noise \
  --output X_train_denoised.npy

# Denoise validation data
uv run -m denoising.denoise \
  --model denoiser_checkpoints/best_model.ckpt \
  --input X_val.npy \
  --add_noise \
  --output X_val_denoised.npy
  
# Denoise test data
uv run -m denoising.denoise \
  --model denoiser_checkpoints/best_model.ckpt \
  --input X_val.npy \
  --output X_val_denoised.npy
```

**Arguments:**
- `--model`: Path to the trained denoising model checkpoint
- `--noisy`: Path to input noisy numpy file
- `--denoised`: Path to output denoised numpy file

### 4. Train the Estimator

Train a cosmological parameter estimator on the denoised images:

```bash
uv run -m estimation.train \
  --x_train X_train_denoised.npy \
  --x_val X_val_denoised.npy \
  --y_train y_train.npy \
  --y_val y_val.npy \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4
```

**Arguments:**
- `--x_train`: Path to training images (default: `X_train.npy`)
- `--x_val`: Path to validation images (default: `X_val.npy`)
- `--y_train`: Path to training labels (default: `y_train.npy`)
- `--y_val`: Path to validation labels (default: `y_val.npy`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)

**Output:**
- Model checkpoints and trained ensemble models
- `label_scaler.skops`: Label scaler for preprocessing
- `label_pca.skops`: PCA model for dimensionality reduction

### 5. Perform Inference

Generate predictions on test data:

```bash
uv run -m estimation.predict \
  --x_val X_val_denoised.npy \
  --y_val y_val.npy \
  --x_test X_test_denoised.npy
```

**Arguments:**
- `--x_val`: Path to validation images (default: `X_val.npy`)
- `--y_val`: Path to validation labels (default: `y_val.npy`)
- `--x_test`: Path to test images (default: `X_test.npy`)
