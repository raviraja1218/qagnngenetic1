import os
import sys
import torch
import tensorflow as tf
import numpy as np
import yaml
from pathlib import Path

print("=== QAGNN Setup Verification ===\n")

# Check Python version
print(f"Python: {sys.version[:6]}")

# Check GPU
print("\n1. GPU Verification:")
print(f"   PyTorch CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"   TensorFlow GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# Check project structure
print("\n2. Project Structure:")
required_dirs = ['data/raw', 'data/processed', 'src', 'models', 'results', 'notebooks']
for dir in required_dirs:
    path = Path(dir)
    if path.exists():
        print(f"   ✓ {dir}")
    else:
        print(f"   ✗ {dir} - MISSING")

# Check config file
print("\n3. Configuration:")
config_path = Path('config.yaml')
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"   ✓ config.yaml loaded")
    print(f"   Project: {config.get('project', {}).get('name', 'N/A')}")
else:
    print(f"   ✗ config.yaml missing")

# Check environment
print("\n4. Environment:")
print(f"   Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'Not set')}")
print(f"   Working dir: {os.getcwd()}")

print("\n=== Setup Complete ===")
if torch.cuda.is_available():
    print("✅ Ready for GPU-accelerated computation!")
else:
    print("⚠️  GPU not detected - check NVIDIA driver")
