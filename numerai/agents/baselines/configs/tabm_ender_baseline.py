# TabM baseline config for target_ender prediction
# ICLR 2025: Parameter-efficient ensemble MLP

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
        'type': 'NumeraiTabM',
        'x_groups': ['features', 'era', 'benchmark_models'],
        'params': {
            'k': 32,                    # Ensemble size
            'n_blocks': 3,              # Number of hidden layers
            'd_block': 256,             # Hidden dimension
            'dropout': 0.1,
            'activation': 'silu',
            'scaling_init': 'normal',   # or 'random-signs'
            'learning_rate': 0.001,
            'batch_size': 4096,
            'epochs': 100,
            'patience': 10,
            'weight_decay': 1e-5,
            'mixed_precision': True,
        },
    },
    'output': {
        'output_dir': 'baselines',
        'results_name': 'tabm_ender_baseline',
    },
    'preprocessing': {'missing_value': 2.0, 'nan_missing_all_twos': False},
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
