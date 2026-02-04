"""Optuna-based neural architecture search for Numerai.

Searches over MLP, ResNet, and FT-Transformer architectures with
hyperparameter tuning optimized for RTX 5070 Ti (16GB VRAM).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna

from gpu_neural_search.data_loader import (
    compute_per_era_corr,
    load_numerai_data,
    split_by_era,
)


# ---------------------------------------------------------------------------
# Search spaces per architecture
# ---------------------------------------------------------------------------

def _suggest_mlp(trial: optuna.Trial, quick: bool = False) -> dict:
    n_layers = trial.suggest_int("mlp_n_layers", 2, 6)
    first_dim = trial.suggest_categorical("mlp_first_dim", [128, 256, 512, 1024])
    shrink = trial.suggest_categorical("mlp_shrink", ["constant", "halving", "gradual"])

    dims = [first_dim]
    for i in range(1, n_layers):
        if shrink == "halving":
            dims.append(max(32, dims[-1] // 2))
        elif shrink == "gradual":
            dims.append(max(32, int(dims[-1] * 0.75)))
        else:
            dims.append(first_dim)

    return {
        "model_type": "NumeraiMLP",
        "params": {
            "hidden_dims": dims,
            "dropout": trial.suggest_float("mlp_dropout", 0.0, 0.5),
            "activation": trial.suggest_categorical("mlp_activation", ["silu", "gelu", "relu"]),
            "use_batchnorm": trial.suggest_categorical("mlp_batchnorm", [True, False]),
            "use_skip": trial.suggest_categorical("mlp_skip", [True, False]),
            "input_dropout": trial.suggest_float("mlp_input_dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("mlp_batch_size", [2048, 4096, 8192]),
            "weight_decay": trial.suggest_float("mlp_wd", 1e-6, 1e-3, log=True),
            "epochs": 30 if quick else trial.suggest_int("mlp_epochs", 50, 150),
            "patience": 5 if quick else 10,
            "mixed_precision": True,
        },
    }


def _suggest_resnet(trial: optuna.Trial, quick: bool = False) -> dict:
    return {
        "model_type": "NumeraiResNet",
        "params": {
            "hidden_dim": trial.suggest_categorical("resnet_dim", [128, 256, 512]),
            "n_blocks": trial.suggest_int("resnet_n_blocks", 2, 8),
            "dropout": trial.suggest_float("resnet_dropout", 0.0, 0.5),
            "activation": trial.suggest_categorical("resnet_activation", ["silu", "gelu", "relu"]),
            "use_batchnorm": trial.suggest_categorical("resnet_batchnorm", [True, False]),
            "input_dropout": trial.suggest_float("resnet_input_dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("resnet_lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("resnet_batch_size", [2048, 4096, 8192]),
            "weight_decay": trial.suggest_float("resnet_wd", 1e-6, 1e-3, log=True),
            "epochs": 30 if quick else trial.suggest_int("resnet_epochs", 50, 150),
            "patience": 5 if quick else 10,
            "mixed_precision": True,
        },
    }


def _suggest_ft_transformer(trial: optuna.Trial, quick: bool = False) -> dict:
    d_model = trial.suggest_categorical("ft_d_model", [64, 96, 128, 192])
    n_heads = trial.suggest_categorical("ft_n_heads", [4, 8])
    # Ensure d_model is divisible by n_heads
    while d_model % n_heads != 0:
        n_heads = n_heads // 2

    return {
        "model_type": "NumeraiFTTransformer",
        "params": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": trial.suggest_int("ft_n_layers", 1, 4),
            "d_ff": trial.suggest_categorical("ft_d_ff", [256, 512, 768]),
            "dropout": trial.suggest_float("ft_dropout", 0.0, 0.4),
            "attention_dropout": trial.suggest_float("ft_attn_dropout", 0.0, 0.3),
            "input_dropout": trial.suggest_float("ft_input_dropout", 0.0, 0.2),
            "learning_rate": trial.suggest_float("ft_lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("ft_batch_size", [1024, 2048, 4096]),
            "weight_decay": trial.suggest_float("ft_wd", 1e-6, 1e-3, log=True),
            "epochs": 20 if quick else trial.suggest_int("ft_epochs", 30, 100),
            "patience": 5 if quick else 10,
            "mixed_precision": True,
        },
    }


_ARCH_SUGGESTERS = {
    "MLP": _suggest_mlp,
    "ResNet": _suggest_resnet,
    "FTTransformer": _suggest_ft_transformer,
}


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _create_objective(
    train_df,
    val_df,
    features: list[str],
    target_col: str,
    quick: bool = False,
):
    """Create an Optuna objective that trains a model and evaluates per-era corr."""

    def objective(trial: optuna.Trial) -> float:
        # Select architecture
        arch = trial.suggest_categorical("architecture", list(_ARCH_SUGGESTERS.keys()))
        config = _ARCH_SUGGESTERS[arch](trial, quick=quick)

        model_type = config["model_type"]
        params = config["params"]

        # Build model
        if model_type == "NumeraiMLP":
            from agents.code.modeling.models.nn_mlp import NumeraiMLP
            model = NumeraiMLP(feature_cols=features, **params)
        elif model_type == "NumeraiResNet":
            from agents.code.modeling.models.nn_resnet import NumeraiResNet
            model = NumeraiResNet(feature_cols=features, **params)
        elif model_type == "NumeraiFTTransformer":
            from agents.code.modeling.models.nn_ft_transformer import NumeraiFTTransformer
            model = NumeraiFTTransformer(feature_cols=features, **params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train
        import pandas as pd
        t0 = time.time()
        try:
            model.fit(train_df[features], train_df[target_col])
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
                print(f"  Trial {trial.number}: OOM - pruning")
                raise optuna.TrialPruned()
            raise
        train_time = time.time() - t0

        # Predict on validation
        val_preds = model.predict(val_df[features])
        val_df_copy = val_df[["era", target_col]].copy()
        val_df_copy["prediction"] = val_preds

        # Compute per-era correlation
        era_corrs = compute_per_era_corr(val_df_copy, "prediction", target_col)
        mean_corr = float(era_corrs.mean())
        std_corr = float(era_corrs.std())
        sharpe = mean_corr / (std_corr + 1e-9)

        # Log metrics
        trial.set_user_attr("mean_corr", mean_corr)
        trial.set_user_attr("std_corr", std_corr)
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("train_time", train_time)
        trial.set_user_attr("architecture", arch)
        trial.set_user_attr("model_config", config)

        print(f"  Trial {trial.number} [{arch}]: mean_corr={mean_corr:.5f}, sharpe={sharpe:.3f}, time={train_time:.1f}s")

        # Clean up GPU memory
        del model
        import torch
        torch.cuda.empty_cache()

        return mean_corr

    return objective


# ---------------------------------------------------------------------------
# Main search runner
# ---------------------------------------------------------------------------

def run_search(
    n_trials: int = 50,
    target_col: str = "target_ender_20",
    feature_set: str = "medium",
    data_version: str = "v5.2",
    output_dir: Path = Path("gpu_neural_search/results"),
    study_name: str = "numerai_nn_search",
    storage: str | None = None,
    resume: bool = False,
    quick: bool = False,
):
    """Run the full architecture search."""
    import torch

    print("=" * 70)
    print("  Numerai GPU Neural Architecture Search")
    print("=" * 70)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  WARNING: No GPU detected. Running on CPU (slow).")

    print(f"  Target: {target_col}")
    print(f"  Feature set: {feature_set}")
    print(f"  Trials: {n_trials}")
    print(f"  Quick mode: {quick}")
    print("=" * 70)

    # Load data
    downsample = 8 if quick else 4
    df, features, target_col = load_numerai_data(
        data_version=data_version,
        feature_set=feature_set,
        target_col=target_col,
        downsample=downsample,
    )

    # Split
    train_df, val_df = split_by_era(df, features, target_col)

    # Create study
    sampler = optuna.samplers.TPESampler(seed=1337, multivariate=True)
    if resume and storage:
        study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            load_if_exists=resume,
        )

    # Run
    objective = _create_objective(train_df, val_df, features, target_col, quick=quick)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Report results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  SEARCH COMPLETE")
    print("=" * 70)

    best = study.best_trial
    print(f"\n  Best trial: #{best.number}")
    print(f"  Best mean_corr: {best.value:.6f}")
    print(f"  Architecture: {best.user_attrs.get('architecture', 'N/A')}")
    print(f"  Sharpe: {best.user_attrs.get('sharpe', 0):.3f}")
    print(f"  Train time: {best.user_attrs.get('train_time', 0):.1f}s")

    # Save best config
    best_config = best.user_attrs.get("model_config", {})
    best_config["search_metrics"] = {
        "mean_corr": best.value,
        "sharpe": best.user_attrs.get("sharpe", 0),
        "std_corr": best.user_attrs.get("std_corr", 0),
    }
    best_config["search_params"] = {
        "target_col": target_col,
        "feature_set": feature_set,
        "data_version": data_version,
        "n_trials": n_trials,
    }

    config_path = output_dir / "best_config.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2, default=str)
    print(f"\n  Best config saved to: {config_path}")

    # Save all trials summary
    trials_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                "number": t.number,
                "value": t.value,
                "architecture": t.user_attrs.get("architecture", ""),
                "sharpe": t.user_attrs.get("sharpe", 0),
                "train_time": t.user_attrs.get("train_time", 0),
                "params": t.params,
            })

    trials_path = output_dir / "all_trials.json"
    with open(trials_path, "w") as f:
        json.dump(trials_data, f, indent=2, default=str)
    print(f"  All trials saved to: {trials_path}")

    # Generate pipeline config for the best model
    _generate_pipeline_config(best_config, target_col, feature_set, data_version, output_dir)

    # Top 5 summary
    print("\n  Top 5 trials:")
    sorted_trials = sorted(trials_data, key=lambda x: x["value"], reverse=True)
    for i, t in enumerate(sorted_trials[:5]):
        print(f"    {i+1}. [{t['architecture']}] corr={t['value']:.5f} sharpe={t['sharpe']:.3f} time={t['train_time']:.0f}s")

    print(f"\n  To train the best model on full data, use:")
    print(f"    PYTHONPATH=numerai python -m agents.code.modeling --config {output_dir / 'best_pipeline_config.py'}")


def _generate_pipeline_config(
    best_config: dict,
    target_col: str,
    feature_set: str,
    data_version: str,
    output_dir: Path,
):
    """Generate a pipeline-compatible config file from the search result."""
    model_type = best_config.get("model_type", "NumeraiMLP")
    params = best_config.get("params", {})

    config_str = f"""# Auto-generated config from neural architecture search
# Best model: {model_type}
# Mean corr: {best_config.get('search_metrics', {}).get('mean_corr', 'N/A')}

CONFIG = {{
    'data': {{
        'data_version': '{data_version}',
        'embargo_eras': 13,
        'era_col': 'era',
        'feature_set': '{feature_set}',
        'target_col': '{target_col}',
    }},
    'model': {{
        'type': '{model_type}',
        'x_groups': ['features', 'era', 'benchmark_models'],
        'params': {repr(params)},
    }},
    'output': {{
        'output_dir': '{output_dir}',
        'results_name': 'best_nn_ender',
    }},
    'preprocessing': {{'missing_value': 2.0, 'nan_missing_all_twos': False}},
    'training': {{
        'cv': {{
            'embargo': 13,
            'enabled': True,
            'min_train_size': 0,
            'mode': 'expanding',
            'n_splits': 5,
        }},
    }},
}}
"""
    config_path = output_dir / "best_pipeline_config.py"
    with open(config_path, "w") as f:
        f.write(config_str)
    print(f"  Pipeline config saved to: {config_path}")
