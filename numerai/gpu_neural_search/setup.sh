#!/usr/bin/env bash
# Setup script for GPU Neural Architecture Search
# Optimized for RTX 5070 Ti with CUDA 12.x
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  Numerai GPU Neural Architecture Search"
echo "  Setup for RTX 5070 Ti"
echo "=============================================="

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed."
fi

# Create virtual environment
VENV_DIR="${PROJECT_DIR}/.venv-gpu"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at ${VENV_DIR} ..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Install PyTorch with CUDA 12.4 support (compatible with RTX 5070 Ti)
echo ""
echo "Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
echo ""
echo "Installing dependencies..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Verify installation
echo ""
echo "Verifying PyTorch CUDA installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    # Test mixed precision support
    print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
    print(f'Flash Attention: available with RTX 5070 Ti')
else:
    print('WARNING: CUDA is not available!')
"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  To activate the environment:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "  Quick test (fast search with 5 trials):"
echo "    cd ${PROJECT_DIR}"
echo "    PYTHONPATH=numerai python -m gpu_neural_search --quick --n-trials 5"
echo ""
echo "  Full search (50 trials):"
echo "    cd ${PROJECT_DIR}"
echo "    PYTHONPATH=numerai python -m gpu_neural_search --n-trials 50"
echo ""
echo "  Train best model after search:"
echo "    PYTHONPATH=numerai python -m agents.code.modeling \\"
echo "      --config gpu_neural_search/results/best_pipeline_config.py"
echo "=============================================="
