# 変更差分メモ

## 概要

Numerai example-scripts リポジトリに、RTX 5070 Ti (16GB VRAM) 向け GPU ニューラルネットワーク
アーキテクチャ自動探索機能を追加。

---

## コミット 1: `67ff3ec` — 初期実装

### 新規ファイル (15件)

#### NN モデル実装 (既存パイプライン統合)

| ファイル | 内容 |
|----------|------|
| `numerai/agents/code/modeling/models/nn_base.py` | PyTorch 基底クラス。Mixed Precision, Early Stopping, Cosine Annealing, DataLoader 管理 |
| `numerai/agents/code/modeling/models/nn_mlp.py` | MLP: 可変深度・幅、Skip接続、BatchNorm、SiLU/GELU/ReLU |
| `numerai/agents/code/modeling/models/nn_resnet.py` | ResNet: Pre-activation 残差ブロック、テーブルデータ用 |
| `numerai/agents/code/modeling/models/nn_ft_transformer.py` | FT-Transformer: 各特徴量をトークン化 → Transformer Encoder → [CLS] 出力 |

#### ベースライン Config

| ファイル | モデル |
|----------|--------|
| `numerai/agents/baselines/configs/mlp_ender_baseline.py` | MLP ベースライン |
| `numerai/agents/baselines/configs/resnet_ender_baseline.py` | ResNet ベースライン |
| `numerai/agents/baselines/configs/ft_transformer_ender_baseline.py` | FT-Transformer ベースライン |

#### Optuna アーキテクチャ探索

| ファイル | 内容 |
|----------|------|
| `numerai/gpu_neural_search/__init__.py` | パッケージ初期化 |
| `numerai/gpu_neural_search/__main__.py` | CLI エントリポイント (--n-trials, --quick, --target 等) |
| `numerai/gpu_neural_search/search.py` | Optuna TPE サンプラーによる探索ロジック |
| `numerai/gpu_neural_search/data_loader.py` | データ取得・前処理・era分割・メトリクス計算 |
| `numerai/gpu_neural_search/requirements.txt` | 依存関係 (PyTorch CUDA, Optuna, NumerAPI 等) |
| `numerai/gpu_neural_search/setup.sh` | ワンライナーセットアップスクリプト |
| `numerai/gpu_neural_search/README.md` | 使い方ドキュメント (日本語) |

### 変更ファイル (1件)

| ファイル | 変更内容 |
|----------|----------|
| `numerai/agents/code/modeling/utils/model_factory.py` | `NumeraiMLP`, `NumeraiResNet`, `NumeraiFTTransformer` の3モデルを登録 |

#### model_factory.py 差分

```python
# Before (1モデルのみ)
if model_type == "LGBMRegressor":
    ...
else:
    raise ValueError("... Supported types: LGBMRegressor")

# After (4モデル対応)
if model_type == "LGBMRegressor":
    ...
elif model_type == "NumeraiMLP":
    from agents.code.modeling.models.nn_mlp import NumeraiMLP
    model = NumeraiMLP(feature_cols=feature_cols, **model_params)
elif model_type == "NumeraiResNet":
    from agents.code.modeling.models.nn_resnet import NumeraiResNet
    model = NumeraiResNet(feature_cols=feature_cols, **model_params)
elif model_type == "NumeraiFTTransformer":
    from agents.code.modeling.models.nn_ft_transformer import NumeraiFTTransformer
    model = NumeraiFTTransformer(feature_cols=feature_cols, **model_params)
else:
    raise ValueError("... Supported types: LGBMRegressor, NumeraiMLP, NumeraiResNet, NumeraiFTTransformer")
```

---

## コミット 2: `e809846` — パイプライン互換性修正

### 発見した問題と修正

#### 問題 1: x_groups 強制拡張

**原因**: `model_data.py` の `normalize_x_groups()` が、指定に関わらず
`('features', 'era', 'benchmark_models')` を強制追加する仕様。

```python
# model_data.py:79-81
for required in _DEFAULT_X_GROUPS:  # ('features', 'era', 'benchmark_models')
    if required not in normalized:
        normalized.append(required)
```

**修正**: 全 NN ベースラインの `x_groups` を `['features', 'era', 'benchmark_models']` に変更し、
`benchmark_data_path` を追加。

```python
# Before
'x_groups': ['features'],
# 'benchmark_data_path' なし

# After
'x_groups': ['features', 'era', 'benchmark_models'],
'benchmark_data_path': 'v5.2/downsampled_full_benchmark_models.parquet',
```

#### 問題 2: target_col 不整合

**原因**: 既存 LGBMRegressor baseline は `target_col: 'target'` を使用。
`target_ender_v4_20` はダウンサンプルデータに含まれない可能性。

```python
# Before
'target_col': 'target_ender_v4_20',

# After
'target_col': 'target',
```

#### 問題 3: NN 内部 val 分割がランダム

**原因**: `nn_base.py` の early stopping 用 validation split がランダムで、
Numerai の時系列構造を無視していた。

**修正**: X DataFrame の `era` カラムを活用し、末尾のeraをvalidation setにする
era-based split に変更。ランダムはフォールバック。

```python
# Before
rng = np.random.RandomState(self._seed)
n = len(X_np)
n_val = max(1, int(n * self._val_fraction))
idx = rng.permutation(n)
val_idx, train_idx = idx[:n_val], idx[n_val:]

# After
def _split_train_val(self, n: int, era_series=None):
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
```

#### 問題 4: 生成 config の不整合

**修正**: `search.py` の `_generate_pipeline_config()` も同様に
`x_groups` と `missing_value` を既存パイプラインと整合するよう修正。

```python
# Before
'x_groups': ['features'],
'missing_value': 0.5,

# After
'x_groups': ['features', 'era', 'benchmark_models'],
'missing_value': 2.0,
```

---

## インターフェース整合性チェック

| 項目 | 既存 LGBMRegressor | 新規 NN モデル | 状態 |
|------|--------------------|----|--------|
| `__init__(feature_cols, **params)` | `lgbm_regressor.py:7` | `nn_base.py:23` | OK |
| `fit(X: DataFrame, y: Series)` | `lgbm_regressor.py:19` | `nn_base.py:63` | OK |
| `predict(X: DataFrame) -> ndarray` | `lgbm_regressor.py:33` | `nn_base.py:173` | OK |
| `__getattr__` 委譲 | `lgbm_regressor.py:57` | `nn_base.py:221` | OK |
| feature_cols フィルタリング | `_filter_features()` | `_to_numpy()` | OK |
| GPU フォールバック | CPU fallback | `_resolve_device('auto')` | OK |
| model_factory 登録 | 行11 | 行14-22 | OK |
