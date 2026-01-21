#!/bin/bash
# Setup script for UCSD cluster - installs OpenPI + LIBERO properly
# Run this ONCE when you start a new pod

set -e  # Exit on error

echo "=============================================="
echo "RoboDeploy Cluster Setup"
echo "=============================================="

# 1. Create a fresh conda env with pinned versions
echo ""
echo "[1/6] Creating conda environment..."
conda create -n robodeploy python=3.11 -y
conda activate robodeploy

# 2. Install numpy with correct version FIRST (before anything else)
echo ""
echo "[2/6] Installing numpy (pinned for numba compatibility)..."
pip install "numpy>=1.24,<2.0"

# 3. Install PyTorch
echo ""
echo "[3/6] Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install LIBERO dependencies
echo ""
echo "[4/6] Installing LIBERO and dependencies..."
pip install robosuite mujoco bddl h5py imageio

# Clone and install LIBERO
if [ ! -d "$HOME/LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/LIBERO
fi
# Create missing __init__.py if needed
touch ~/LIBERO/libero/__init__.py 2>/dev/null || true
pip install -e ~/LIBERO

# 5. Install OpenPI
echo ""
echo "[5/6] Installing OpenPI..."
if [ ! -d "$HOME/openpi" ]; then
    git clone https://github.com/Physical-Intelligence/openpi.git ~/openpi
fi
cd ~/openpi
pip install -e .

# Apply transformers patches (required by OpenPI)
echo "Applying OpenPI transformers patches..."
cp -r third_party/transformers_replace/* $(python -c "import transformers; print(transformers.__path__[0])")/

# 6. Install remaining deps
echo ""
echo "[6/6] Installing remaining dependencies..."
pip install omegaconf einops safetensors huggingface_hub

# Download PI0.5 checkpoint
echo ""
echo "Downloading PI0.5 checkpoint (this takes a few minutes)..."
python -c "
from openpi.shared import download
checkpoint_dir = download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')
print(f'Checkpoint downloaded to: {checkpoint_dir}')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To run evaluation:"
echo "  conda activate robodeploy"
echo "  export MUJOCO_GL=osmesa"
echo "  export PYOPENGL_PLATFORM=osmesa"
echo "  python ~/robodeploy-continual/scripts/eval_baseline.py --task=libero_spatial --n-episodes=3"
