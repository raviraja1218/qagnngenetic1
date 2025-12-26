#!/bin/bash
echo "=== PHASE 1 MONITORING ==="
echo "Monitoring started at: $(date)"
echo ""

while true; do
    clear
    echo "========================================="
    echo "PHASE 1 MONITOR - $(date)"
    echo "========================================="
    
    # GPU Status
    echo ""
    echo "=== GPU STATUS ==="
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv
    
    # CPU Status
    echo ""
    echo "=== CPU STATUS ==="
    top -bn1 | grep "Cpu(s)" | awk '{print "CPU Usage: " $2 "%"}'
    
    # Memory Status
    echo ""
    echo "=== MEMORY STATUS ==="
    free -h | awk 'NR==2{printf "Memory: %s/%s (%.2f%%)\n", $3,$2,$3*100/$2}'
    
    # Disk Status
    echo ""
    echo "=== DISK STATUS ==="
    df -h ~ | awk 'NR==2{printf "Home: %s/%s (%.1f%%)\n", $3,$2,$5}'
    du -sh ~/projects/qagnn/data/
    
    # Process Check
    echo ""
    echo "=== PROCESS CHECK ==="
    if pgrep -f "python.*circuit" > /dev/null; then
        echo "✓ Simulation process is RUNNING"
    else
        echo "✗ Simulation process is NOT running"
    fi
    
    # Check log file size
    echo ""
    echo "=== LOG STATUS ==="
    if [ -f "results/logs/phase1/simulation_progress.log" ]; then
        lines=$(wc -l < results/logs/phase1/simulation_progress.log)
        echo "Progress log lines: $lines"
    fi
    
    echo ""
    echo "========================================="
    echo "Press Ctrl+C to stop monitoring"
    echo "Next update in 30 seconds..."
    sleep 30
done
