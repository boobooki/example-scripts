"""Base class for all PyTorch neural network models in the Numerai pipeline.

Provides common training loop, GPU management, early stopping, and
integration with the existing fit/predict interface.
Optimized for RTX 5070 Ti (16GB VRAM, CUDA).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class NumeraiNNBase:
    """Base wrapper for PyTorch models compatible with the Numerai pipeline.

    Subclasses must implement ``_build_network(n_features) -> nn.Module``.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        learning_rate: float = 1e-3,
        batch_size: int = 4096,
        epochs: int = 100,
        patience: int = 10,
        weight_decay: float = 1e-5,
        device: str = "auto",
        num_workers: int = 4,
        pin_memory: bool = True,
        mixed_precision: bool = True,
        val_fraction: float = 0.1,
        seed: int = 1337,
        verbose: bool = True,
        **kwargs,
    ):
        self._feature_cols = feature_cols
        self._lr = learning_rate
        self._batch_size = batch_size
        self._epochs = epochs
        self._patience = patience
        self._weight_decay = weight_decay
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._mixed_precision = mixed_precision
        self._val_fraction = val_fraction
        self._seed = seed
        self._verbose = verbose
        self._extra_params = kwargs

        self._device = self._resolve_device(device)
        self._network: nn.Module | None = None
        self._n_features: int | None = None

    # ------------------------------------------------------------------
    # Public interface (fit / predict)
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "NumeraiNNBase":
        # Extract era info before feature filtering (for era-based val split)
        era_series = None
        if hasattr(X, "columns") and "era" in X.columns:
            era_series = X["era"]

        X_np, y_np = self._prepare_data(X, y)
        self._n_features = X_np.shape[1]

        # Train / validation split (era-based if possible, random fallback)
        train_idx, val_idx = self._split_train_val(len(X_np), era_series)

        train_ds = TensorDataset(
            torch.as_tensor(X_np[train_idx], dtype=torch.float32),
            torch.as_tensor(y_np[train_idx], dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.as_tensor(X_np[val_idx], dtype=torch.float32),
            torch.as_tensor(y_np[val_idx], dtype=torch.float32),
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=self._batch_size * 2,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

        # Build network & optimizer
        self._network = self._build_network(self._n_features).to(self._device)
        optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._epochs, eta_min=self._lr * 0.01
        )
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda", enabled=self._mixed_precision and self._device.type == "cuda")

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self._epochs + 1):
            # --- Train ---
            self._network.train()
            train_loss = 0.0
            n_batches = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self._device, non_blocking=True), yb.to(self._device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self._mixed_precision and self._device.type == "cuda"):
                    pred = self._network(xb).squeeze(-1)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train = train_loss / max(n_batches, 1)

            # --- Validate ---
            self._network.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self._device, non_blocking=True), yb.to(self._device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=self._mixed_precision and self._device.type == "cuda"):
                        pred = self._network(xb).squeeze(-1)
                        loss = criterion(pred, yb)
                    val_loss += loss.item()
                    n_val_batches += 1

            avg_val = val_loss / max(n_val_batches, 1)

            if self._verbose and epoch % 5 == 0:
                print(f"  Epoch {epoch:>3d}/{self._epochs} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in self._network.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    if self._verbose:
                        print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.6f})")
                    break

        if best_state is not None:
            self._network.load_state_dict(best_state)
            self._network.to(self._device)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_np = self._to_numpy(X)
        self._network.eval()
        preds = []
        ds = TensorDataset(torch.as_tensor(X_np, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=self._batch_size * 2, shuffle=False,
                        num_workers=self._num_workers, pin_memory=self._pin_memory)
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(self._device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=self._mixed_precision and self._device.type == "cuda"):
                    pred = self._network(xb).squeeze(-1)
                preds.append(pred.float().cpu().numpy())
        return np.concatenate(preds)

    # ------------------------------------------------------------------
    # Abstract method
    # ------------------------------------------------------------------

    def _build_network(self, n_features: int) -> nn.Module:
        raise NotImplementedError("Subclasses must implement _build_network()")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_train_val(self, n: int, era_series=None):
        """Split indices into train/val.  Uses era-based split when era info
        is available (last eras as val) to respect the temporal structure of
        Numerai data.  Falls back to random split otherwise."""
        n_val = max(1, int(n * self._val_fraction))
        if era_series is not None:
            # Era-based split: use last eras as validation
            eras = era_series.values
            unique_eras = sorted(set(eras), key=lambda e: (int(e) if str(e).isdigit() else e))
            n_val_eras = max(1, int(len(unique_eras) * self._val_fraction))
            val_eras = set(unique_eras[-n_val_eras:])
            val_mask = np.array([e in val_eras for e in eras])
            val_idx = np.where(val_mask)[0]
            train_idx = np.where(~val_mask)[0]
            if len(val_idx) > 0 and len(train_idx) > 0:
                return train_idx, val_idx
        # Random fallback
        rng = np.random.RandomState(self._seed)
        idx = rng.permutation(n)
        return idx[n_val:], idx[:n_val]

    def _prepare_data(self, X, y):
        X_np = self._to_numpy(X)
        y_np = y.values.astype(np.float32) if hasattr(y, "values") else np.asarray(y, dtype=np.float32)
        return X_np, y_np

    def _to_numpy(self, X):
        if self._feature_cols and hasattr(X, "columns"):
            X = X[self._feature_cols]
        if hasattr(X, "values"):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._network is not None:
            return getattr(self._network, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
