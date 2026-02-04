# Auto-generated config from neural architecture search
# Best model: NumeraiResNet
# Mean corr: 0.009539785839474568

CONFIG = {
    'data': {
        'data_version': 'v5.2',
        'embargo_eras': 13,
        'era_col': 'era',
        'feature_set': 'medium',
        'target_col': 'target_ender_20',
    },
    'model': {
        'type': 'NumeraiResNet',
        'x_groups': ['features', 'era', 'benchmark_models'],
        'params': {'hidden_dim': 128, 'n_blocks': 8, 'dropout': 0.316774302970865, 'activation': 'relu', 'use_batchnorm': False, 'input_dropout': 0.07084476270060593, 'learning_rate': 0.0007236314951725256, 'batch_size': 4096, 'weight_decay': 0.00011244996290932906, 'epochs': 30, 'patience': 5, 'mixed_precision': True},
    },
    'output': {
        'output_dir': 'gpu_neural_search\results',
        'results_name': 'best_nn_ender',
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
