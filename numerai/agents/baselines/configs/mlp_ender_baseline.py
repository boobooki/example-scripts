# MLP baseline config for target_ender prediction
# Optimized for RTX 5070 Ti (16GB VRAM)

CONFIG = {
    'data': {
        'data_version': 'v5.2',
        'embargo_eras': 13,
        'era_col': 'era',
        'feature_set': 'medium',
        'target_col': 'target',
        'full_data_path': 'v5.2/downsampled_full.parquet',
        'benchmark_data_path': 'v5.2/downsampled_full_benchmark_models.parquet',
    },
    'model': {
        'type': 'NumeraiMLP',
        'x_groups': ['features', 'era', 'benchmark_models'],
        'params': {
            'hidden_dims': [512, 256, 128],
            'dropout': 0.15,
            'activation': 'silu',
            'use_batchnorm': True,
            'use_skip': True,
            'input_dropout': 0.05,
            'learning_rate': 0.001,
            'batch_size': 4096,
            'epochs': 100,
            'patience': 10,
            'weight_decay': 1e-5,
            'mixed_precision': True,
            'device': 'auto',
        },
    },
    'output': {
        'output_dir': 'baselines',
        'results_name': 'mlp_ender_baseline',
    },
    'preprocessing': {'missing_value': 0.5, 'nan_missing_all_twos': False},
    'training': {
        'cv': {
            'embargo': 13,
            'enabled': True,
            'min_train_size': 0,
            'mode': 'expanding',
            'n_splits': 5,
        },
    },
}
