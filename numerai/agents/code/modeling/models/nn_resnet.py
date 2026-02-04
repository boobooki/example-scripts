"""ResNet-style model for Numerai tabular data.

Pre-activation residual blocks with configurable depth and width.
Designed for GPU training on RTX 5070 Ti.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from agents.code.modeling.models.nn_base import NumeraiNNBase


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, activation: str, use_batchnorm: bool):
        super().__init__()
        layers: list[nn.Module] = []
        # Pre-activation residual block
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(_get_activation(activation))
        layers.append(nn.Linear(dim, dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(_get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim, dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _ResNetNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_blocks: int,
        dropout: float,
        activation: str,
        use_batchnorm: bool,
        input_dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        layers.append(nn.Linear(n_features, hidden_dim))
        for _ in range(n_blocks):
            layers.append(_ResBlock(hidden_dim, dropout, activation, use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(_get_activation(activation))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NumeraiResNet(NumeraiNNBase):
    """ResNet wrapper compatible with the Numerai pipeline."""

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
        activation: str = "silu",
        use_batchnorm: bool = True,
        input_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(feature_cols=feature_cols, **kwargs)
        self._hidden_dim = hidden_dim
        self._n_blocks = n_blocks
        self._dropout = dropout
        self._activation = activation
        self._use_batchnorm = use_batchnorm
        self._input_dropout = input_dropout

    def _build_network(self, n_features: int) -> nn.Module:
        return _ResNetNetwork(
            n_features=n_features,
            hidden_dim=self._hidden_dim,
            n_blocks=self._n_blocks,
            dropout=self._dropout,
            activation=self._activation,
            use_batchnorm=self._use_batchnorm,
            input_dropout=self._input_dropout,
        )


def _get_activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(),
    }.get(name.lower(), nn.SiLU())
