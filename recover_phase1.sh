#!/bin/bash
# Phase 1 Recovery Script
# Use this if WSL crashes or session is lost

echo "=== PHASE 1 RECOVERY ==="
echo "Checking system status..."

# Check if conda environment is active
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Activating conda environment..."
    conda activate qagnn
fi

# Check if in correct directory
if [[ ! -f "verify_setup.py" ]]; then
    echo "Changing to project directory..."
    cd ~/projects/qagnn
fi

# Check what's already done
echo "Checking progress..."
if [ -f "data/raw/batch_0000_0999.h5" ]; then
    echo "✅ Batch 1 already completed"
    BATCHES_DONE=1
else
    echo "❌ Batch 1 not completed"
    BATCHES_DONE=0
fi

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check disk space
echo "Checking disk space..."
df -h ~ | grep -v snap

# Offer options
echo ""
echo "Recovery options:"
echo "1. Resume from last checkpoint"
echo "2. Start fresh"
echo "3. Just verify existing data"
echo "4. Create backup and exit"
echo ""
read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "Resuming from last checkpoint..."
        if [ $BATCHES_DONE -eq 1 ]; then
            echo "Starting Batch 2..."
            python run_batch_2.py 2>&1 | tee results/logs/phase1/batch_2.log
        else
            echo "Starting Batch 1..."
            python run_batch_1.py 2>&1 | tee results/logs/phase1/batch_1.log
        fi
        ;;
    2)
        echo "Starting fresh..."
        echo "Backing up existing data..."
        mkdir -p ~/project_backups/qagnn_crash_recovery_$(date +%Y%m%d_%H%M%S)
        cp -r data/raw/*.h5 ~/project_backups/qagnn_crash_recovery_*/ 2>/dev/null || true
        
        echo "Starting Batch 1..."
        python run_batch_1.py 2>&1 | tee results/logs/phase1/batch_1.log
        ;;
    3)
        echo "Verifying existing data..."
        python verify_batch.py 2>&1 | tee results/logs/phase1/verification.log
        ;;
    4)
        echo "Creating backup..."
        mkdir -p ~/project_backups/qagnn_phase1_backup_$(date +%Y%m%d_%H%M%S)
        cp -r data/ ~/project_backups/qagnn_phase1_backup_*/
        cp -r results/logs/phase1/ ~/project_backups/qagnn_phase1_backup_*/
        echo "Backup created. Safe to exit."
        ;;
    *)
        echo "Invalid option. Exiting."
        ;;
esac
