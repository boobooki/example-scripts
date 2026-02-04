"""Entry point for the GPU neural architecture search.

Usage:
    cd example-scripts
    PYTHONPATH=numerai python -m gpu_neural_search [OPTIONS]

Options:
    --n-trials        Number of Optuna trials (default: 50)
    --target          Target column (default: target_ender_v4_20)
    --feature-set     Feature set: small, medium, all (default: medium)
    --data-version    Dataset version (default: v5.2)
    --output-dir      Output directory (default: gpu_neural_search/results)
    --study-name      Optuna study name (default: numerai_nn_search)
    --storage         Optuna storage URL for distributed search (default: None)
    --resume          Resume existing study (flag)
    --quick           Quick mode: fewer epochs, smaller data (flag)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="GPU Neural Network Architecture Search for Numerai"
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna search trials")
    parser.add_argument("--target", type=str, default="target_ender_v4_20", help="Target column name")
    parser.add_argument("--feature-set", type=str, default="medium", choices=["small", "medium", "all"], help="Feature set size")
    parser.add_argument("--data-version", type=str, default="v5.2", help="Numerai data version")
    parser.add_argument("--output-dir", type=str, default="gpu_neural_search/results", help="Output directory")
    parser.add_argument("--study-name", type=str, default="numerai_nn_search", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///study.db)")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    args = parser.parse_args()

    from gpu_neural_search.search import run_search

    run_search(
        n_trials=args.n_trials,
        target_col=args.target,
        feature_set=args.feature_set,
        data_version=args.data_version,
        output_dir=Path(args.output_dir),
        study_name=args.study_name,
        storage=args.storage,
        resume=args.resume,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
