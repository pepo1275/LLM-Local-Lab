#!/bin/bash
# Post-Benchmark Processing Hook
# Runs after benchmark completion to save results and trigger analysis

set -e

echo "=== Post-Benchmark Processing ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

# Get the most recent benchmark result file
RESULTS_DIR="C:/Users/Gamer/Dev/LLM-Local-Lab/benchmarks/results/raw"
LATEST_RESULT=$(ls -t "$RESULTS_DIR"/*.json 2>/dev/null | head -1)

if [ -z "$LATEST_RESULT" ]; then
    echo "WARNING: No benchmark results found in $RESULTS_DIR"
    exit 0
fi

echo "Latest result: $LATEST_RESULT"
FILENAME=$(basename "$LATEST_RESULT")

# Extract key metrics for quick summary
echo -e "\n[1/4] Extracting metrics..."
if command -v python &> /dev/null; then
    python -c "
import json
import sys

try:
    with open('$LATEST_RESULT', 'r') as f:
        data = json.load(f)

    print('Model:', data.get('model', 'Unknown'))
    print('Throughput:', data.get('throughput', 'N/A'), 'tokens/s')
    print('VRAM Peak:', data.get('vram_peak', 'N/A'), 'GB')
    print('Duration:', data.get('duration', 'N/A'), 's')
except Exception as e:
    print('Could not parse metrics:', str(e), file=sys.stderr)
" || echo "Could not extract metrics"
fi

# Generate metrics snapshot
echo -e "\n[2/4] Generating metrics snapshot..."
SNAPSHOT_FILE="$RESULTS_DIR/${FILENAME%.json}_snapshot.txt"
cat > "$SNAPSHOT_FILE" <<EOF
Benchmark Snapshot
==================
File: $FILENAME
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
System: AMD 9950X + Dual RTX 5090 (64GB VRAM)

GPU Status at completion:
$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv)

EOF

echo "Snapshot saved: $SNAPSHOT_FILE"

# Log completion
echo -e "\n[3/4] Logging benchmark completion..."
LOG_FILE="C:/Users/Gamer/Dev/LLM-Local-Lab/benchmarks/benchmark_history.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') | COMPLETED | $FILENAME" >> "$LOG_FILE"

# Suggest next action
echo -e "\n[4/4] Next steps:"
echo "  - Review results: cat $LATEST_RESULT"
echo "  - Generate report: @documentation-writer document the results"
echo "  - Analyze metrics: @benchmark-analyst analyze latest results"
echo ""

echo "=== Post-Benchmark Processing Complete ==="
exit 0
