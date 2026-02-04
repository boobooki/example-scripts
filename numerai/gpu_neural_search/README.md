# GPU Neural Architecture Search for Numerai

RTX 5070 Ti (16GB VRAM) に最適化された、Numerai トーナメント向けニューラルネットワーク
アーキテクチャ自動探索ツール。

## 概要

Optuna を使って3種類のニューラルネットワークアーキテクチャを自動探索し、
`target_ender` の予測精度が最も高いモデルを見つけます。

### 探索対象アーキテクチャ

| モデル | 説明 | 特徴 |
|--------|------|------|
| **NumeraiMLP** | Multi-Layer Perceptron | Skip接続、BatchNorm、可変深度・幅 |
| **NumeraiResNet** | ResNet (tabular) | Pre-activation残差ブロック |
| **NumeraiFTTransformer** | Feature Tokenizer + Transformer | 各特徴量をトークン化してAttention |

### 探索されるハイパーパラメータ

- アーキテクチャ選択 (MLP / ResNet / FT-Transformer)
- レイヤー数、隠れ層の次元
- Dropout率、活性化関数
- 学習率、バッチサイズ、Weight decay
- BatchNorm、Skip接続の有無

## セットアップ

```bash
# リポジトリのルートから
cd example-scripts

# 自動セットアップ (venv作成 + PyTorch CUDA + 依存関係)
bash numerai/gpu_neural_search/setup.sh
```

### 手動セットアップ

```bash
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate

# PyTorch with CUDA 12.4 (RTX 5070 Ti対応)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# その他の依存関係
pip install -r numerai/gpu_neural_search/requirements.txt
```

## 使い方

### クイックテスト (5トライアル)

```bash
cd example-scripts
source .venv-gpu/bin/activate
PYTHONPATH=numerai python -m gpu_neural_search --quick --n-trials 5
```

### フルサーチ (50トライアル)

```bash
PYTHONPATH=numerai python -m gpu_neural_search --n-trials 50
```

### カスタム設定

```bash
PYTHONPATH=numerai python -m gpu_neural_search \
  --n-trials 100 \
  --target target_ender_v4_20 \
  --feature-set medium \
  --output-dir gpu_neural_search/results \
  --study-name my_search
```

### 分散探索 (複数GPUマシン)

```bash
# SQLiteストレージで共有
PYTHONPATH=numerai python -m gpu_neural_search \
  --n-trials 100 \
  --storage sqlite:///optuna_study.db \
  --study-name shared_search

# 別のマシンから同じstudyに参加
PYTHONPATH=numerai python -m gpu_neural_search \
  --n-trials 100 \
  --storage sqlite:///optuna_study.db \
  --study-name shared_search \
  --resume
```

## 探索後のワークフロー

### 1. ベストモデルをフルデータで学習

探索完了後、`results/best_pipeline_config.py` が自動生成されます。

```bash
PYTHONPATH=numerai python -m agents.code.modeling \
  --config gpu_neural_search/results/best_pipeline_config.py
```

### 2. 個別モデルを直接実行

```bash
# MLP ベースライン
PYTHONPATH=numerai python -m agents.code.modeling \
  --config agents/baselines/configs/mlp_ender_baseline.py

# ResNet ベースライン
PYTHONPATH=numerai python -m agents.code.modeling \
  --config agents/baselines/configs/resnet_ender_baseline.py

# FT-Transformer ベースライン
PYTHONPATH=numerai python -m agents.code.modeling \
  --config agents/baselines/configs/ft_transformer_ender_baseline.py
```

## 出力ファイル

```
gpu_neural_search/results/
├── best_config.json           # ベストモデルの設定
├── best_pipeline_config.py    # パイプライン互換の設定ファイル
└── all_trials.json            # 全トライアルの結果
```

## RTX 5070 Ti 最適化ポイント

- **Mixed Precision (FP16)**: 自動有効化で学習速度2倍、VRAM使用量半減
- **大バッチサイズ**: 16GB VRAMを活かした4096〜8192バッチ
- **Pin Memory**: CPU→GPU転送の高速化
- **Cosine Annealing**: 学習率スケジューリングで収束改善
- **Gradient Clipping**: 安定した学習
- **Early Stopping**: 無駄なエポックを省略
- **OOM対策**: VRAM不足時にトライアルを自動スキップ

## VRAM使用量の目安

| モデル | feature_set=medium | feature_set=all |
|--------|-------------------|-----------------|
| MLP | ~2-4 GB | ~4-8 GB |
| ResNet | ~2-5 GB | ~5-10 GB |
| FT-Transformer | ~4-8 GB | ~10-16 GB |

FT-Transformer は特徴量数に対してメモリ使用量が大きいため、
`feature_set: all` の場合は `d_model` や `n_layers` を小さくしてください。

## プロジェクト構造

```
numerai/
├── gpu_neural_search/           # アーキテクチャ探索ツール
│   ├── __main__.py             # エントリポイント
│   ├── search.py               # Optuna探索ロジック
│   ├── data_loader.py          # データ読み込み
│   ├── setup.sh                # セットアップスクリプト
│   ├── requirements.txt        # 依存関係
│   └── README.md               # このファイル
├── agents/
│   ├── code/modeling/models/
│   │   ├── nn_base.py          # PyTorch基底クラス
│   │   ├── nn_mlp.py           # MLP実装
│   │   ├── nn_resnet.py        # ResNet実装
│   │   └── nn_ft_transformer.py # FT-Transformer実装
│   └── baselines/configs/
│       ├── mlp_ender_baseline.py
│       ├── resnet_ender_baseline.py
│       └── ft_transformer_ender_baseline.py
```
