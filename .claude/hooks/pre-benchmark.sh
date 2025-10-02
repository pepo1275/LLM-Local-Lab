#!/bin/bash
# Pre-Benchmark Validation Hook
# Runs before benchmark execution to ensure system is ready

set -e

echo "=== Pre-Benchmark Validation ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

# Check 1: GPU Utilization (should be <10%)
echo -e "\n[1/5] Checking GPU utilization..."
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')
echo "GPU utilization: $GPU_UTIL"

for util in $GPU_UTIL; do
    if [ "$util" -gt 10 ]; then
        echo "ERROR: GPU utilization too high ($util%). GPUs must be idle (<10%) before benchmark."
        echo "Active processes:"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
        exit 1
    fi
done
echo "OK: GPUs are idle"

# Check 2: GPU Temperature (should be <50C for consistent performance)
echo -e "\n[2/5] Checking GPU temperature..."
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr '\n' ' ')
echo "GPU temperatures: $GPU_TEMP C"

for temp in $GPU_TEMP; do
    if [ "$temp" -gt 50 ]; then
        echo "WARNING: GPU temperature elevated ($temp C). Results may vary due to thermal throttling."
        echo "Consider waiting for GPUs to cool down."
    fi
done
echo "OK: Temperatures acceptable"

# Check 3: Available VRAM
echo -e "\n[3/5] Checking available VRAM..."
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)
echo "Free VRAM: $FREE_VRAM"

# Check 4: Disk space for results (need at least 1GB)
echo -e "\n[4/5] Checking disk space..."
BENCH_DIR="C:/Users/Gamer/Dev/LLM-Local-Lab/benchmarks/results"
if [ -d "$BENCH_DIR" ]; then
    FREE_SPACE=$(df -BG "$BENCH_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
    echo "Free disk space: ${FREE_SPACE}GB"

    if [ "$FREE_SPACE" -lt 1 ]; then
        echo "WARNING: Low disk space (<1GB). Clean up old results or increase storage."
    fi
fi
echo "OK: Sufficient disk space"

# Check 5: Clear CUDA cache
echo -e "\n[5/5] Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')" 2>/dev/null || {
    echo "WARNING: Could not clear CUDA cache. PyTorch may not be available."
}

echo -e "\n=== Validation Complete: System Ready for Benchmark ==="
echo ""

exit 0
