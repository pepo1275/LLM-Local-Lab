#!/bin/bash
# Auto-Commit Results Hook
# Automatically commits benchmark results with standardized format

set -e

echo "=== Auto-Commit Results ==="

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Not in a git repository. Skipping auto-commit."
    exit 0
fi

# Get the most recent benchmark result
RESULTS_DIR="C:/Users/Gamer/Dev/LLM-Local-Lab/benchmarks/results/raw"
LATEST_RESULT=$(ls -t "$RESULTS_DIR"/*.json 2>/dev/null | head -1)

if [ -z "$LATEST_RESULT" ]; then
    echo "No benchmark results to commit."
    exit 0
fi

FILENAME=$(basename "$LATEST_RESULT")
TIMESTAMP=$(echo "$FILENAME" | cut -d'_' -f1-2)
MODEL_NAME=$(echo "$FILENAME" | cut -d'_' -f3- | sed 's/.json$//')

# Check if this file is already committed
if git ls-files --error-unmatch "$LATEST_RESULT" > /dev/null 2>&1; then
    if ! git diff --quiet "$LATEST_RESULT"; then
        echo "File has uncommitted changes."
    else
        echo "File already committed. Skipping."
        exit 0
    fi
fi

# Extract key metric for commit message
THROUGHPUT="unknown"
if command -v python &> /dev/null; then
    THROUGHPUT=$(python -c "
import json
try:
    with open('$LATEST_RESULT', 'r') as f:
        data = json.load(f)
    print(data.get('throughput', 'unknown'))
except:
    print('unknown')
" 2>/dev/null)
fi

# Create commit message
COMMIT_MSG="[BENCHMARK]: $MODEL_NAME - $THROUGHPUT tokens/s

Benchmark results for $MODEL_NAME
Timestamp: $TIMESTAMP
Throughput: $THROUGHPUT tokens/s

Results file: benchmarks/results/raw/$FILENAME

Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

# Add and commit the result file
echo "Committing: $FILENAME"
git add "$LATEST_RESULT"

# Also commit the snapshot if it exists
SNAPSHOT_FILE="${LATEST_RESULT%.json}_snapshot.txt"
if [ -f "$SNAPSHOT_FILE" ]; then
    git add "$SNAPSHOT_FILE"
fi

# Create commit
git commit -m "$COMMIT_MSG" || {
    echo "WARNING: Commit failed. Files may already be staged."
    exit 0
}

echo "Successfully committed benchmark results."
echo ""

# Ask if user wants to push
echo "To push these results to remote:"
echo "  git push origin \$(git branch --show-current)"
echo ""

exit 0
