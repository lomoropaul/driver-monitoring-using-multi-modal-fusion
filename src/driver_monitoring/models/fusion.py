"""EEG + face fusion models (proposal section 3.7.2).

* Early fusion:  ``y = f([h_eeg; h_face])``, branch embeddings concatenated
  and passed to a shared classifier.
* Late fusion:   ``p = α p_eeg + (1 - α) p_face`` with α tuned on validation;
  it needs no new network, see :mod:`driver_monitoring.training.late_fusion`.
* Hybrid fusion: both embeddings are projected to a common size and an
  attention gate learns, per window, how much to trust each modality, so a
  degraded modality (a dark frame, a noisy channel) can be down-weighted.

Both trainable models use *modality dropout* during training: with
probability ``modality_dropout`` one modality's embedding is zeroed for a
sample (never both), so the model cannot rely on a single stream and learns to
cope when one fails (proposal section 3.8, step 6).
"""

from __future__ import annotations

import torch
from torch import nn

from .branches import EEGNet, FaceNet


class _FusionBase(nn.Module):
    def __init__(self, eeg_kwargs: dict | None = None, face_kwargs: dict | None = None,
                 modality_dropout: float = 0.15):
        super().__init__()
        self.eeg = EEGNet(**(eeg_kwargs or {}))
        self.face = FaceNet(**(face_kwargs or {}))
        self.modality_dropout = modality_dropout

    def _embeddings(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        h_e, h_f = self.eeg.embed(batch), self.face.embed(batch)
        if self.training and self.modality_dropout > 0:
            b = h_e.shape[0]
            r = torch.rand(b, device=h_e.device)
            drop_e = (r < self.modality_dropout).float().unsqueeze(1)
            drop_f = ((r >= self.modality_dropout) & (r < 2 * self.modality_dropout)).float().unsqueeze(1)
            h_e, h_f = h_e * (1 - drop_e), h_f * (1 - drop_f)
        return h_e, h_f

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.embed(batch))


class EarlyFusion(_FusionBase):
    def __init__(self, eeg_kwargs=None, face_kwargs=None, hidden: int = 64, dropout: float = 0.3,
                 modality_dropout: float = 0.15, n_classes: int = 2):
        super().__init__(eeg_kwargs, face_kwargs, modality_dropout)
        self.embed_dim = hidden
        self.mlp = nn.Sequential(nn.Linear(self.eeg.embed_dim + self.face.embed_dim, hidden),
                                 nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden, n_classes)

    def embed(self, batch: dict) -> torch.Tensor:
        return self.mlp(torch.cat(self._embeddings(batch), dim=1))

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)


class HybridFusion(_FusionBase):
    """Attention-gated intermediate fusion.

    ``z_m = P_m h_m``; ``w = softmax(g([z_eeg; z_face]))``; ``z = Σ w_m z_m``.
    The per-window gate weights are kept in ``self.last_weights`` for analysis.
    """

    def __init__(self, eeg_kwargs=None, face_kwargs=None, dim: int = 64, dropout: float = 0.3,
                 modality_dropout: float = 0.15, n_classes: int = 2):
        super().__init__(eeg_kwargs, face_kwargs, modality_dropout)
        self.proj_eeg = nn.Sequential(nn.Linear(self.eeg.embed_dim, dim), nn.LayerNorm(dim), nn.ReLU())
        self.proj_face = nn.Sequential(nn.Linear(self.face.embed_dim, dim), nn.LayerNorm(dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 2))
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = dim
        self.classifier = nn.Linear(dim, n_classes)
        self.last_weights: torch.Tensor | None = None

    def embed(self, batch: dict) -> torch.Tensor:
        h_e, h_f = self._embeddings(batch)
        z_e, z_f = self.proj_eeg(h_e), self.proj_face(h_f)
        w = torch.softmax(self.gate(torch.cat([z_e, z_f], dim=1)), dim=1)
        self.last_weights = w.detach()
        return self.dropout(w[:, :1] * z_e + w[:, 1:] * z_f)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)
