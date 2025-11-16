import math

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR


# PyTorch Lightning CNN module for parameter estimation
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return nn.functional.avg_pool2d(
            x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))
        ).pow(1.0 / p)

    def __repr__(self):
        return (
                self.__class__.__name__
                + "("
                + "p="
                + "{:.4f}".format(self.p.data.tolist()[0])
                + ", "
                + "eps="
                + str(self.eps)
                + ")"
        )


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ParameterEstimator(pl.LightningModule):
    def __init__(self, cnn, feature_size, num_targets, learning_rate=2e-4, weight_decay=1e-4):
        super(ParameterEstimator, self).__init__()
        self.save_hyperparameters(ignore=["cnn"])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.cnn = cnn

        self.pooling = GeM()

        # MLP outputs 256 features
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(feature_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(256, 256),
        )

        # self.pos_embedding = PositionalEncoding(d_model=256, dropout=0.1, max_len=32)
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=4, batch_first=True
        )

        self.fc_out = nn.Linear(256, num_targets)
        self.dropout = nn.Dropout(0.3, inplace=False)

    def forward_features(self, x):
        batch_size, num_patches, channels, *_ = x.shape
        img = x.contiguous().view(-1, channels, 88, 88)

        x = self.cnn.forward_features(img)
        x = self.pooling(x)
        x = self.mlp(x)
        x = x.reshape(batch_size, num_patches, -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        # x = self.pos_embedding(x.transpose(0, 1)).transpose(0, 1)
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)

        # Aggregate patches (mean pooling across patches)
        x = attn_output.mean(1)

        # Final output layer
        x = self.fc_out(x)  # (batch_size, num_targets)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=1200)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        preds = [model(x) for model in self.models]
        preds_stack = torch.stack(preds, dim=0)
        mean_pred = torch.mean(preds_stack, dim=0)
        return mean_pred
