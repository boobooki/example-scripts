# RealMLP baseline config for target_ender prediction
# NeurIPS 2024: Better by Default - bag-of-tricks MLP

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
        'type': 'NumeraiRealMLP',
        'x_groups': ['features', 'era', 'benchmark_models'],
        'params': {
            'hidden_dims': [256, 256, 256],  # 3x256 as per paper
            'activation': 'selu',             # SELU is default for RealMLP
            'dropout': 0.0,                   # SELU works best without dropout
            'use_ntp': True,                  # Neural tangent parametrization
            'use_smooth_clip': True,          # Smooth clipping for outliers
            'clip_value': 3.0,
            'use_robust_scaling': True,       # Robust scaling preprocessing
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
        'results_name': 'realmlp_ender_baseline',
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
