# FT-Transformer baseline config for target_ender prediction
# Optimized for RTX 5070 Ti (16GB VRAM)
# NOTE: FT-Transformer is memory-intensive with many features.
#       Use feature_set 'medium' or 'small' to avoid OOM.

CONFIG = {
    'data': {
        'data_version': 'v5.2',
        'embargo_eras': 13,
        'era_col': 'era',
        'feature_set': 'medium',
        'target_col': 'target_ender_v4_20',
        'full_data_path': 'v5.2/downsampled_full.parquet',
    },
    'model': {
        'type': 'NumeraiFTTransformer',
        'x_groups': ['features'],
        'params': {
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 3,
            'd_ff': 512,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'input_dropout': 0.05,
            'learning_rate': 0.0005,
            'batch_size': 2048,
            'epochs': 80,
            'patience': 10,
            'weight_decay': 1e-5,
            'mixed_precision': True,
            'device': 'auto',
        },
    },
    'output': {
        'output_dir': 'baselines',
        'results_name': 'ft_transformer_ender_baseline',
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
