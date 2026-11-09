"""Single-modality branches (proposal section 3.7.1).

Every model takes a batch dict and exposes two steps so the same branches can
be reused by the fusion models later:

* ``embed(batch) -> h``    feature vector per window, size ``embed_dim``
* ``head(h) -> logits``    two-class logits (alert, vigilance decline)

``forward(batch)`` = ``head(embed(batch))``.
"""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    """Compact spatio-temporal CNN for EEG (Lawhern et al., 2018).

    Input ``batch["eeg"]``: ``[B, channels, samples]`` (z-scored).
    Temporal conv -> depthwise spatial conv -> separable conv, as in EEGNet-8,2.
    """

    def __init__(self, n_channels: int = 5, n_samples: int = 512, f1: int = 8, depth: int = 2,
                 f2: int = 16, kernel: int = 64, dropout: float = 0.25, n_classes: int = 2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * depth, f1 * depth, (1, 16), padding=(0, 8), groups=f1 * depth, bias=False),
            nn.Conv2d(f1 * depth, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            self.embed_dim = self._features(torch.zeros(1, n_channels, n_samples)).shape[1]
        self.classifier = nn.Linear(self.embed_dim, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x.unsqueeze(1))).flatten(1)

    def embed(self, batch: dict) -> torch.Tensor:
        return self._features(batch["eeg"])

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.embed(batch))


def _sep_conv(cin: int, cout: int, stride: int) -> nn.Sequential:
    """MobileNet-style depthwise separable convolution block."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin),
        nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class FaceNet(nn.Module):
    """Light CNN per frame + GRU over the window's frames + hand-crafted features.

    Inputs:
    * ``batch["crops"]``: ``[B, frames, H, W]`` eye/mouth crops scaled to 0-1
    * ``batch["face_feats"]``: ``[B, n_feats]`` EAR, MAR, PERCLOS, ... (z-scored)
    """

    def __init__(self, n_feats: int = 10, width: int = 16, hidden: int = 64,
                 dropout: float = 0.3, n_classes: int = 2, use_feats: bool = True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, width, 3, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            _sep_conv(width, width * 2, 2),
            _sep_conv(width * 2, width * 4, 2),
            _sep_conv(width * 4, width * 4, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.gru = nn.GRU(width * 4, hidden, batch_first=True)
        self.use_feats = use_feats
        feat_dim = 16 if use_feats else 0
        self.feat_mlp = nn.Sequential(nn.Linear(n_feats, feat_dim), nn.ReLU()) if use_feats else None
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = hidden + feat_dim
        self.classifier = nn.Linear(self.embed_dim, n_classes)

    def embed(self, batch: dict) -> torch.Tensor:
        x = batch["crops"]
        b, t, h, w = x.shape
        frames = self.cnn(x.reshape(b * t, 1, h, w)).reshape(b, t, -1)
        _, last = self.gru(frames)
        z = last[-1]
        if self.use_feats:
            z = torch.cat([z, self.feat_mlp(batch["face_feats"])], dim=1)
        return self.dropout(z)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.embed(batch))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, **kwargs) -> nn.Module:
    if name == "eeg":
        return EEGNet(**kwargs)
    if name == "face":
        return FaceNet(**kwargs)
    from .fusion import EarlyFusion, HybridFusion

    if name == "early":
        return EarlyFusion(**kwargs)
    if name == "hybrid":
        return HybridFusion(**kwargs)
    raise ValueError(f"Unknown model {name!r} (expected 'eeg', 'face', 'early' or 'hybrid')")
