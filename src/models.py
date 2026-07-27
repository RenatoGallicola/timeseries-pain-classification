"""Model definitions.

Three architectures reach the final submission:

* ``CNN_GRU_Classifier`` — the 3-class workhorse, instantiated twice as models
  A and B with different window sizes.
* ``InceptionTime`` — model C, a multi-scale convolutional network run on the
  enriched 92-feature representation.
* ``CNN_GRU_Binary`` — a low_pain/high_pain specialist whose logits are fused
  into the ensemble by the gate.

All are trained from scratch, as the competition rules require.
"""
from __future__ import annotations

import torch
from torch import nn


class CNN_GRU_Classifier(nn.Module):
    """Conv1d feature extractor followed by a GRU encoder.

    The convolutions pick up local, short-range motion patterns; the GRU carries
    them across the window. Pooling over time is either attention-weighted or a
    plain mean — the submitted models A and B use the mean (`use_attention=False`).
    """

    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout_rate: float = 0.3,
                 bidirectional: bool = True, use_attention: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        gru_input_size = 128

        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, gru_input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(gru_input_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_output_size = hidden_size * (2 if bidirectional else 1)

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(gru_output_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1, bias=False),
            )

        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):                 # x: (B, T, F)
        x = x.permute(0, 2, 1)            # (B, F, T)
        x = self.cnn(x)                   # (B, C, T)
        x = x.permute(0, 2, 1)            # (B, T, C)
        out, _ = self.gru(x)              # (B, T, H)

        if self.use_attention:
            weights = torch.softmax(self.attention(out), dim=1)
            x = torch.sum(out * weights, dim=1)
        else:
            x = out.mean(dim=1)

        return self.fc(x)


class InceptionBlock(nn.Module):
    """One Inception module: three parallel convolutions with kernels 10/20/40
    plus a max-pool + 1x1 branch, concatenated over the channel axis.

    The point of the different kernel widths is to cover several temporal scales
    at once, which the single-scale CNN front-end of the GRU models cannot do.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.b1 = nn.Conv1d(in_channels, out_channels, kernel_size=10, padding="same")
        self.b2 = nn.Conv1d(in_channels, out_channels, kernel_size=20, padding="same")
        self.b3 = nn.Conv1d(in_channels, out_channels, kernel_size=40, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.b4 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same")
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):                 # x: (B, C, T)
        out = torch.cat(
            [self.b1(x), self.b2(x), self.b3(x), self.b4(self.pool(x))], dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    """Model C: three stacked Inception blocks, global average pooling, linear head."""

    def __init__(self, num_features: int, channels: int = 32, num_classes: int = 3):
        super().__init__()
        self.block1 = InceptionBlock(num_features, channels)
        self.block2 = InceptionBlock(channels * 4, channels)
        self.block3 = InceptionBlock(channels * 4, channels)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels * 4, num_classes)

    def forward(self, x):                 # x: (B, T, F)
        x = x.permute(0, 2, 1)            # (B, F, T)
        x = self.block3(self.block2(self.block1(x)))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


class CNN_GRU_Binary(nn.Module):
    """low_pain vs high_pain specialist.

    Same trunk as ``CNN_GRU_Classifier`` so that its logits live on a comparable
    scale, plus a small MLP over the static subject attributes
    (``n_legs``, ``n_hands``, ``n_eyes``) concatenated before the classifier head.
    """

    def __init__(self, input_size_dynamic: int, input_size_static: int = 3,
                 hidden_size: int = 160, num_layers: int = 2,
                 dropout_rate: float = 0.3, bidirectional: bool = True,
                 use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        gru_input_size = 128

        self.cnn = nn.Sequential(
            nn.Conv1d(input_size_dynamic, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, gru_input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(gru_input_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        feat_dim = hidden_size * (2 if bidirectional else 1)

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1, bias=False),
            )

        self.static_mlp = nn.Sequential(
            nn.Linear(input_size_static, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.fc = nn.Sequential(
            nn.Linear(feat_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2),
        )

    def forward(self, x_dyn, x_static):
        x = x_dyn.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)

        if self.use_attention:
            weights = torch.softmax(self.attention(out), dim=1)
            seq_embed = torch.sum(out * weights, dim=1)
        else:
            seq_embed = out.mean(dim=1)

        return self.fc(torch.cat([seq_embed, self.static_mlp(x_static)], dim=1))
