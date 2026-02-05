"""TabM: Parameter-Efficient Ensemble MLP (ICLR 2025).

Efficiently imitates an ensemble of MLPs through weight sharing and
parallel training.  Based on "TabM: Advancing Tabular Deep Learning
With Parameter-Efficient Ensembling" (Gorishniy et al., 2025).

Optimized for RTX 5070 Ti (16GB VRAM).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.code.modeling.models.nn_base import NumeraiNNBase


class _LinearBatchEnsemble(nn.Module):
    """Linear layer with batch ensemble (weight sharing + elementwise scaling).

    Creates k diverse representations through learned scaling vectors before
    the shared linear transformation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        bias: bool = True,
        scaling_init: str = "normal",
    ):
        super().__init__()
        self.k = k
        self.in_features = in_features
        self.out_features = out_features

        # Shared weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(k, out_features))
        else:
            self.register_parameter("bias", None)

        # Per-ensemble scaling vectors (r_i and s_i in the paper)
        self.r = nn.Parameter(torch.empty(k, in_features))  # input scaling
        self.s = nn.Parameter(torch.empty(k, out_features))  # output scaling

        self.scaling_init = scaling_init
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize scaling vectors
        if self.scaling_init == "random-signs":
            # Random ±1
            self.r.data = torch.randint(0, 2, self.r.shape).float() * 2 - 1
            self.s.data = torch.randint(0, 2, self.s.shape).float() * 2 - 1
        else:
            # Normal initialization around 1
            nn.init.normal_(self.r, mean=1.0, std=0.1)
            nn.init.normal_(self.s, mean=1.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, k, in_features) or (batch, in_features)
        if x.dim() == 2:
            # Expand to k copies: (batch, in_features) -> (batch, k, in_features)
            x = x.unsqueeze(1).expand(-1, self.k, -1)

        # Apply input scaling: x * r[i] for each ensemble member
        # x: (batch, k, in_features), r: (k, in_features)
        x = x * self.r.unsqueeze(0)  # (batch, k, in_features)

        # Shared linear transformation
        # (batch, k, in) @ (in, out) -> (batch, k, out)
        out = torch.matmul(x, self.weight.t())

        # Apply output scaling
        out = out * self.s.unsqueeze(0)  # (batch, k, out_features)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)  # (batch, k, out_features)

        return out


class _TabMBlock(nn.Module):
    """Single TabM block with BatchEnsemble linear + activation + dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        dropout: float,
        activation: str,
        scaling_init: str,
    ):
        super().__init__()
        self.linear = _LinearBatchEnsemble(
            in_features, out_features, k, scaling_init=scaling_init
        )
        self.activation = _get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class _TabMNetwork(nn.Module):
    """TabM network: BatchEnsemble MLP with k parallel predictions."""

    def __init__(
        self,
        n_features: int,
        k: int,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str,
        scaling_init: str,
    ):
        super().__init__()
        self.k = k

        layers = []
        prev_dim = n_features
        for _ in range(n_blocks):
            layers.append(
                _TabMBlock(prev_dim, d_block, k, dropout, activation, scaling_init)
            )
            prev_dim = d_block

        self.blocks = nn.Sequential(*layers)
        self.head = _LinearBatchEnsemble(prev_dim, 1, k, scaling_init=scaling_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        x = self.blocks(x)  # (batch, k, d_block)
        x = self.head(x)  # (batch, k, 1)
        return x.squeeze(-1)  # (batch, k)


class NumeraiTabM(NumeraiNNBase):
    """TabM wrapper compatible with the Numerai pipeline.

    TabM efficiently ensembles k MLPs through weight sharing, providing
    regularization and improved generalization with minimal overhead.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        activation: str = "silu",
        scaling_init: str = "normal",
        **kwargs,
    ):
        """Initialize TabM.

        Args:
            k: Ensemble size (number of parallel predictions). Default: 32.
            n_blocks: Number of hidden layers. Default: 3.
            d_block: Hidden dimension. Default: 256.
            dropout: Dropout rate. Default: 0.1.
            activation: Activation function. Default: "silu".
            scaling_init: How to init scaling vectors ("normal" or "random-signs").
        """
        super().__init__(feature_cols=feature_cols, **kwargs)
        self._k = k
        self._n_blocks = n_blocks
        self._d_block = d_block
        self._dropout = dropout
        self._activation = activation
        self._scaling_init = scaling_init

    def _build_network(self, n_features: int) -> nn.Module:
        return _TabMNetwork(
            n_features=n_features,
            k=self._k,
            n_blocks=self._n_blocks,
            d_block=self._d_block,
            dropout=self._dropout,
            activation=self._activation,
            scaling_init=self._scaling_init,
        )

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean loss across ensemble members (required for TabM).

        Each ensemble member is trained independently, then averaged at inference.
        """
        # pred: (batch, k), target: (batch,)
        # Remove extra dimension if present (from network output)
        if pred.dim() == 3:
            pred = pred.squeeze(-1)  # (batch, k)
        # Compute loss for each ensemble member and average
        target_expanded = target.unsqueeze(1).expand_as(pred)  # (batch, k)
        return F.mse_loss(pred, target_expanded)

    def predict(self, X):
        """Average k predictions at inference."""
        import numpy as np

        if self._network is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_np = self._to_numpy(X)
        self._network.eval()

        from torch.utils.data import DataLoader, TensorDataset

        preds = []
        ds = TensorDataset(torch.as_tensor(X_np, dtype=torch.float32))
        dl = DataLoader(
            ds,
            batch_size=self._batch_size * 2,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(self._device, non_blocking=True)
                with torch.amp.autocast(
                    "cuda", enabled=self._mixed_precision and self._device.type == "cuda"
                ):
                    pred = self._network(xb)  # (batch, k)
                    pred_mean = pred.mean(dim=1)  # (batch,)
                preds.append(pred_mean.float().cpu().numpy())
        return np.concatenate(preds)


def _get_activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "elu": nn.ELU(),
        "mish": nn.Mish(),
    }.get(name.lower(), nn.SiLU())
