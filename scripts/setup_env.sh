#!/bin/bash
# Setup script for RoboDeploy Continual Learning
# Run this on your cluster to install dependencies

set -e

echo "=== RoboDeploy Continual Learning Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install LIBERO from source
echo "Installing LIBERO..."
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git

# Install OpenPI from source  
echo "Installing OpenPI..."
pip install git+https://github.com/Physical-Intelligence/openpi.git

# Install LeRobot
echo "Installing LeRobot..."
pip install lerobot

# Download PI0.5 checkpoint
echo "Downloading PI0.5 checkpoint..."
python scripts/download_checkpoint.py

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# Check GPU memory
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {props.name}, {props.total_memory / 1e9:.1f}GB')
"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run baseline eval: python scripts/eval_baseline.py"
echo "3. Or run the full loop: python scripts/run_continual.py"
