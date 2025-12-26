#!/bin/bash
# QAGNN Daily Startup

echo "Starting QAGNN Project..."

# Activate conda environment
conda activate qagnn

# Check GPU
echo "Checking GPU..."
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"

# Open VSCode in project directory
echo "Opening project in VSCode..."
code ~/projects/qagnn

echo "Ready to work!"
