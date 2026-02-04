"""Multi-Layer Perceptron for Numerai tabular data.

Supports configurable depth, width, dropout, batch-norm, skip connections,
and SiLU/GELU/ReLU activations.  Designed for GPU training on RTX 5070 Ti.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from agents.code.modeling.models.nn_base import NumeraiNNBase


class _MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, activation: str, use_batchnorm: bool, use_skip: bool):
        super().__init__()
        self.use_skip = use_skip and (in_dim == out_dim)
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(_get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_skip:
            out = out + x
        return out


class _MLPNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int],
        dropout: float,
        activation: str,
        use_batchnorm: bool,
        use_skip: bool,
        input_dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        prev = n_features
        for dim in hidden_dims:
            layers.append(_MLPBlock(prev, dim, dropout, activation, use_batchnorm, use_skip))
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NumeraiMLP(NumeraiNNBase):
    """MLP wrapper compatible with the Numerai pipeline."""

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "silu",
        use_batchnorm: bool = True,
        use_skip: bool = True,
        input_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(feature_cols=feature_cols, **kwargs)
        self._hidden_dims = hidden_dims or [512, 256, 128]
        self._dropout = dropout
        self._activation = activation
        self._use_batchnorm = use_batchnorm
        self._use_skip = use_skip
        self._input_dropout = input_dropout

    def _build_network(self, n_features: int) -> nn.Module:
        return _MLPNetwork(
            n_features=n_features,
            hidden_dims=self._hidden_dims,
            dropout=self._dropout,
            activation=self._activation,
            use_batchnorm=self._use_batchnorm,
            use_skip=self._use_skip,
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
