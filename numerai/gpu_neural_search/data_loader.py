"""Data loading utilities for the neural architecture search.

Handles downloading and caching Numerai tournament data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_numerai_data(
    data_version: str = "v5.2",
    feature_set: str = "medium",
    target_col: str = "target_ender_20",
    downsample: int = 4,
) -> tuple[pd.DataFrame, list[str], str]:
    """Load Numerai tournament data.

    Returns:
        (dataframe, feature_columns, target_column)
    """
    try:
        from numerapi import NumerAPI
    except ImportError:
        raise ImportError(
            "numerapi is required. Install with: pip install numerapi"
        )

    napi = NumerAPI()

    # Determine data directory
    data_dir = Path(data_version)
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    # Download training data if needed
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        print(f"Downloading {data_version} training data...")
        napi.download_dataset(f"{data_version}/train.parquet", str(train_path))

    # Load features info from features.json
    features_path = data_dir / "features.json"
    if not features_path.exists():
        print(f"Downloading {data_version} features.json...")
        napi.download_dataset(f"{data_version}/features.json", str(features_path))

    with open(features_path) as f:
        feature_metadata = json.load(f)

    if feature_set in feature_metadata["feature_sets"]:
        features = feature_metadata["feature_sets"][feature_set]
    else:
        features = feature_metadata["feature_sets"]["medium"]

    # Load data
    cols_to_load = ["era", target_col] + features
    print(f"Loading data from {train_path} ...")
    df = pd.read_parquet(train_path, columns=cols_to_load)

    # Downsample by era for faster iteration
    if downsample > 1:
        eras = sorted(df["era"].unique())
        selected_eras = eras[::downsample]
        df = df[df["era"].isin(selected_eras)].reset_index(drop=True)
        print(f"Downsampled to every {downsample} eras: {len(selected_eras)} eras, {len(df)} rows")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Fill missing features with 0.5 (Numerai convention: features are [0, 1] range)
    df[features] = df[features].fillna(0.5)

    print(f"Loaded {len(df)} rows, {len(features)} features, target={target_col}")
    return df, features, target_col


def split_by_era(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    val_fraction: float = 0.2,
    embargo: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/validation by era with embargo.

    Returns:
        (train_df, val_df)
    """
    eras = sorted(df["era"].unique())
    n_eras = len(eras)
    n_val = max(1, int(n_eras * val_fraction))

    val_eras = set(eras[-n_val:])
    # Embargo: remove eras close to val boundary
    embargo_start = n_eras - n_val - embargo
    embargo_eras = set(eras[max(0, embargo_start): n_eras - n_val])

    train_mask = ~df["era"].isin(val_eras | embargo_eras)
    val_mask = df["era"].isin(val_eras)

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    print(f"Train: {len(train_df)} rows ({n_eras - n_val - len(embargo_eras)} eras), "
          f"Val: {len(val_df)} rows ({n_val} eras), Embargo: {len(embargo_eras)} eras")

    return train_df, val_df


def compute_numerai_corr(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Numerai-style rank correlation."""
    from scipy import stats
    # Rank predictions and targets
    pred_rank = stats.rankdata(predictions)
    target_rank = stats.rankdata(targets)
    # Standardize
    pred_rank = (pred_rank - pred_rank.mean()) / pred_rank.std()
    target_rank = (target_rank - target_rank.mean()) / target_rank.std()
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])


def compute_per_era_corr(
    df: pd.DataFrame, pred_col: str, target_col: str
) -> pd.Series:
    """Compute per-era Numerai correlation."""
    from scipy import stats

    def _era_corr(group):
        if len(group) < 2:
            return 0.0
        pred = stats.rankdata(group[pred_col])
        tgt = stats.rankdata(group[target_col])
        pred = (pred - pred.mean()) / (pred.std() + 1e-9)
        tgt = (tgt - tgt.mean()) / (tgt.std() + 1e-9)
        return float(np.corrcoef(pred, tgt)[0, 1])

    return df.groupby("era").apply(_era_corr)
