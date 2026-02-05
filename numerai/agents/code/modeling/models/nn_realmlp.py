"""RealMLP: Bag-of-Tricks MLP (NeurIPS 2024).

Implements the "Better by Default" MLP with:
- Robust scaling and smooth clipping
- SELU/Mish activations
- Neural tangent parametrization
- Optimized default hyperparameters

Based on "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on
Tabular Data" (Holzmüller et al., NeurIPS 2024).

Optimized for RTX 5070 Ti (16GB VRAM).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.code.modeling.models.nn_base import NumeraiNNBase


class _SmoothClip(nn.Module):
    """Smooth clipping function to handle outliers.

    Uses a soft approximation: x / (1 + |x/c|) * c
    This smoothly limits values to approximately [-c, c].
    """

    def __init__(self, clip_value: float = 3.0):
        super().__init__()
        self.c = clip_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.abs(x / self.c)) * self.c


class _NTPLinear(nn.Module):
    """Linear layer with Neural Tangent Parametrization.

    Scales weights by 1/sqrt(fan_in) during forward pass for better
    gradient flow and training stability.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.scale = 1.0 / math.sqrt(in_features)
        self._init_weights()

    def _init_weights(self):
        # Initialize with larger variance since we scale down
        nn.init.normal_(self.weight, std=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)


class _RealMLPBlock(nn.Module):
    """Single RealMLP block: Linear + Activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str,
        dropout: float,
        use_ntp: bool,
    ):
        super().__init__()
        if use_ntp:
            self.linear = _NTPLinear(in_features, out_features)
        else:
            self.linear = nn.Linear(in_features, out_features)
        self.activation = _get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


class _RealMLPNetwork(nn.Module):
    """RealMLP network with bag-of-tricks."""

    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int],
        activation: str,
        dropout: float,
        use_ntp: bool,
        use_smooth_clip: bool,
        clip_value: float,
    ):
        super().__init__()
        self.smooth_clip = _SmoothClip(clip_value) if use_smooth_clip else nn.Identity()

        layers = []
        prev_dim = n_features
        for dim in hidden_dims:
            layers.append(_RealMLPBlock(prev_dim, dim, activation, dropout, use_ntp))
            prev_dim = dim

        self.blocks = nn.Sequential(*layers)

        # Output layer (no activation, no dropout)
        if use_ntp:
            self.head = _NTPLinear(prev_dim, 1)
        else:
            self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.smooth_clip(x)
        x = self.blocks(x)
        return self.head(x)


class NumeraiRealMLP(NumeraiNNBase):
    """RealMLP wrapper compatible with the Numerai pipeline.

    RealMLP is an improved MLP with carefully tuned defaults and
    "bag-of-tricks" modifications for better tabular data performance.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        hidden_dims: list[int] | None = None,
        activation: str = "selu",
        dropout: float = 0.0,
        use_ntp: bool = True,
        use_smooth_clip: bool = True,
        clip_value: float = 3.0,
        use_robust_scaling: bool = True,
        **kwargs,
    ):
        """Initialize RealMLP.

        Args:
            hidden_dims: Hidden layer sizes. Default: [256, 256, 256].
            activation: Activation function ("selu", "mish", "gelu"). Default: "selu".
            dropout: Dropout rate. Default: 0.0 (SELU works best without dropout).
            use_ntp: Use Neural Tangent Parametrization. Default: True.
            use_smooth_clip: Apply smooth clipping to inputs. Default: True.
            clip_value: Smooth clip threshold. Default: 3.0.
            use_robust_scaling: Use robust scaling in preprocessing. Default: True.
        """
        super().__init__(feature_cols=feature_cols, **kwargs)
        self._hidden_dims = hidden_dims or [256, 256, 256]
        self._activation = activation
        self._dropout = dropout
        self._use_ntp = use_ntp
        self._use_smooth_clip = use_smooth_clip
        self._clip_value = clip_value
        self._use_robust_scaling = use_robust_scaling

        # For robust scaling
        self._medians: np.ndarray | None = None
        self._iqrs: np.ndarray | None = None

    def _build_network(self, n_features: int) -> nn.Module:
        return _RealMLPNetwork(
            n_features=n_features,
            hidden_dims=self._hidden_dims,
            activation=self._activation,
            dropout=self._dropout,
            use_ntp=self._use_ntp,
            use_smooth_clip=self._use_smooth_clip,
            clip_value=self._clip_value,
        )

    def _prepare_data(self, X, y):
        """Override to apply robust scaling."""
        X_np = self._to_numpy(X)

        if self._use_robust_scaling:
            # Compute robust scaling parameters from training data
            self._medians = np.nanmedian(X_np, axis=0)
            q75 = np.nanpercentile(X_np, 75, axis=0)
            q25 = np.nanpercentile(X_np, 25, axis=0)
            self._iqrs = q75 - q25
            self._iqrs = np.where(self._iqrs < 1e-8, 1.0, self._iqrs)  # Avoid div by zero
            X_np = (X_np - self._medians) / self._iqrs

        y_np = y.values.astype(np.float32) if hasattr(y, "values") else np.asarray(y, dtype=np.float32)
        return X_np.astype(np.float32), y_np

    def _to_numpy(self, X):
        """Override to apply robust scaling at inference."""
        if self._feature_cols and hasattr(X, "columns"):
            X = X[self._feature_cols]
        X_np = X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)

        if self._use_robust_scaling and self._medians is not None:
            X_np = (X_np - self._medians) / self._iqrs

        return X_np


def _get_activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "selu": nn.SELU(),
        "mish": nn.Mish(),
        "elu": nn.ELU(),
    }.get(name.lower(), nn.SELU())
