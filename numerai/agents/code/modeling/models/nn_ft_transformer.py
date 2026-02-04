"""Feature Tokenizer + Transformer for Numerai tabular data.

Each feature is projected into an embedding space, then processed by a
standard Transformer encoder.  Includes a [CLS] token for aggregation.
Designed for GPU training on RTX 5070 Ti.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from agents.code.modeling.models.nn_base import NumeraiNNBase


class _FeatureTokenizer(nn.Module):
    """Projects each scalar feature into a d_model-dimensional embedding."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_features, d_model))
        self.biases = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.zeros_(self.biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weights + self.biases


class _FTTransformerNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        attention_dropout: float,
        input_dropout: float,
    ):
        super().__init__()
        self.tokenizer = _FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        if input_dropout > 0:
            self.input_drop = nn.Dropout(input_dropout)
        else:
            self.input_drop = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tokenize features
        tokens = self.tokenizer(x)  # (B, F, D)
        tokens = self.input_drop(tokens)
        # Prepend [CLS] token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, F+1, D)
        # Transformer encoding
        out = self.transformer(tokens)
        # Use [CLS] output for prediction
        cls_out = out[:, 0]
        return self.head(cls_out)


class NumeraiFTTransformer(NumeraiNNBase):
    """FT-Transformer wrapper compatible with the Numerai pipeline.

    Note: For very high-dimensional feature sets (>2000 features),
    the attention mechanism can be memory-intensive.  Use ``feature_set: medium``
    or ``feature_set: small`` in configs, or reduce ``d_model`` and ``n_layers``.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        input_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(feature_cols=feature_cols, **kwargs)
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._d_ff = d_ff
        self._dropout = dropout
        self._attention_dropout = attention_dropout
        self._input_dropout = input_dropout

    def _build_network(self, n_features: int) -> nn.Module:
        return _FTTransformerNetwork(
            n_features=n_features,
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_layers=self._n_layers,
            d_ff=self._d_ff,
            dropout=self._dropout,
            attention_dropout=self._attention_dropout,
            input_dropout=self._input_dropout,
        )
