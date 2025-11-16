import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl


class DoubleConv(nn.Module):
    """Two consecutive conv layers with BatchNorm and ReLU"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle potential size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetAutoencoder(nn.Module):
    """
    U-Net architecture for denoising autoencoder on full images.
    Input: (B, 1, H, W) - Variable sized images
    Output: features (B, feature_dim), reconstructed (B, 1, H, W)
    """

    def __init__(self, in_channels=1, feature_dim=512):
        super(UNetAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Encoder (downsampling path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Bottleneck - adaptive pooling to handle variable input sizes
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

        # Store spatial sizes for reconstruction
        self.spatial_sizes = []

        # Decoder starting point - needs to be adaptive
        self.decoder_start = nn.Sequential(
            nn.Conv2d(feature_dim, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.GELU()
        )

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.dropout = nn.Dropout(p=0.2)

        # Output layer
        self.outc = nn.Conv2d(64, in_channels, kernel_size=1)
        self.output_act = nn.Identity()

    def forward(self, x):
        # Store input size
        input_size = x.shape[2:]

        # Encoder path - store intermediate feature maps for skip connections
        x1 = self.inc(x)  # (B, 64, H, W)
        # x1 = self.dropout(x1)
        x2 = self.down1(x1)  # (B, 128, H/2, W/2)
        # x2 = self.dropout(x2)
        x3 = self.down2(x2)  # (B, 256, H/4, W/4)
        # x3 = self.dropout(x3)
        x4 = self.down3(x3)  # (B, 512, H/8, W/8)
        x5 = self.down4(x4)  # (B, 1024, H/16, W/16)

        # Bottleneck
        features = self.bottleneck(x5)  # (B, feature_dim, 1, 1)

        # features = self.dropout(features)

        # Flatten features for PCA
        features_flat = features.view(features.size(0), -1)  # (B, feature_dim)

        # Decoder path with skip connections
        # Start from bottleneck and upsample to match x5 size
        # x_up = self.decoder_start(features)  # (B, 1024, 1, 1)
        # x_up = F.interpolate(x_up, size=x5.shape[2:], mode='bilinear', align_corners=False)

        x_up = self.up1(x5, x4)  # (B, 512, H/8, W/8)
        x_up = self.up2(x_up, x3)  # (B, 256, H/4, W/4)
        x_up = self.up3(x_up, x2)  # (B, 128, H/2, W/2)
        x_up = self.up4(x_up, x1)  # (B, 64, H, W)

        # Final reconstruction
        reconstructed = self.output_act(self.outc(x_up))  # (B, 1, H, W)

        return features_flat, reconstructed


class DenoisingUNet(pl.LightningModule):
    def __init__(self, feature_dim=512, learning_rate=2e-4, weight_decay=1e-4):
        super(DenoisingUNet, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.autoencoder = UNetAutoencoder(in_channels=1, feature_dim=feature_dim)

    def forward(self, x):
        """Extract features and reconstruct"""
        features, reconstructed = self.autoencoder(x)
        return features, reconstructed

    def training_step(self, batch, batch_idx):
        x_noisy, x_clean, *_ = batch

        # Forward pass
        features, reconstructed = self(x_noisy)

        # Denoising loss (MSE between reconstruction and clean image)
        loss = F.mse_loss(reconstructed, x_clean)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_noisy, x_clean, *_ = batch

        # Forward pass
        features, reconstructed = self(x_noisy)

        # Denoising loss
        loss = F.mse_loss(reconstructed, x_clean)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
