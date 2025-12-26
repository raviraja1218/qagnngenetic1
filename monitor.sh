#!/bin/bash
# QAGNN System Monitor

echo "=== System Status ==="
echo "Time: $(date)"

echo -e "\n--- GPU ---"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

echo -e "\n--- CPU ---"
top -bn1 | grep "Cpu(s)"

echo -e "\n--- Memory ---"
free -h

echo -e "\n--- Disk ---"
df -h ~/projects/qagnn

echo -e "\n--- Conda Environment ---"
conda env list | grep "*"
