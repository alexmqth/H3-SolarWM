#!/bin/bash
# Wait for scheduled-sampling experiments to complete and run benchmarks
# Usage: bash wait_and_benchmark.sh

BASE_DIR="/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/2026-10-02-scheduled-sampling"
BENCHMARK_SCRIPT="/home/lpeng/code/mq_PubDataset/GWM/H3-World/code/causal/benchmark.py"

echo "Waiting for all experiments to complete..."
echo "Started at: $(date)"
echo ""

# Wait for all experiments to finish
while true; do
    all_done=true

    for exp in baseline_clean_only fixed_mix_0.5 curriculum_0.0_to_0.5; do
        json_file="$BASE_DIR/$exp/training.json"
        if [ -f "$json_file" ]; then
            status=$(python3 -c "import json; print(json.load(open('$json_file')).get('status', 'unknown'))" 2>/dev/null)
            if [ "$status" != "complete" ]; then
                all_done=false
            fi
        else
            all_done=false
        fi
    done

    if $all_done; then
        echo "All experiments complete at: $(date)"
        break
    fi

    # Print status every 5 minutes
    bash /home/qma/work/GWM/monitor_scheduled_sampling.sh
    sleep 300
done

echo ""
echo "=========================================="
echo "Running benchmarks for all three experiments..."
echo "=========================================="
echo ""

source /home/lpeng/code/mq_PubDataset/GWM/.venvs/h3world/bin/activate

# Find available GPU
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 40000 {print $1; exit}')
if [ -z "$GPU" ]; then
    echo "ERROR: No GPU with >40GB free memory available"
    exit 1
fi

echo "Using GPU $GPU for benchmarks"
echo ""

# Run benchmark for each experiment
for exp in baseline_clean_only fixed_mix_0.5 curriculum_0.0_to_0.5; do
    echo "=== Benchmarking $exp ==="

    CUDA_VISIBLE_DEVICES=$GPU python $BENCHMARK_SCRIPT \
        --checkpoint "$BASE_DIR/$exp/adapter.safetensors" \
        --out-dir "$BASE_DIR/${exp}_benchmark" \
        --actions W A D \
        --frames 39 \
        --anchor-mode dynamic_last_frame_dual \
        --history-protocol cached \
        --scheduler-steps 8 \
        --device cuda:0

    if [ $? -eq 0 ]; then
        echo "✓ Benchmark complete: $exp"
        echo "  Flow results: $BASE_DIR/${exp}_benchmark/flow.json"
    else
        echo "✗ Benchmark failed: $exp"
    fi
    echo ""
done

echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo ""
echo "To view results:"
echo "  cat $BASE_DIR/*/training.json | jq '.train_after, .validation_after'"
echo "  cat $BASE_DIR/*_benchmark/flow.json | jq '.A_minus_D_horizontal'"
