#!/bin/bash
# Monitor scheduled-sampling experiments
# Usage: bash monitor_scheduled_sampling.sh

BASE_DIR="/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/2026-10-02-scheduled-sampling"

echo "=========================================="
echo "Scheduled-Sampling Experiments Monitor"
echo "Time: $(date)"
echo "=========================================="
echo ""

for exp in baseline_clean_only fixed_mix_0.5 curriculum_0.0_to_0.5; do
    echo "=== $exp ==="

    json_file="$BASE_DIR/$exp/training.json"
    if [ -f "$json_file" ]; then
        status=$(python3 -c "import json; d=json.load(open('$json_file')); print(d.get('status', 'unknown'))" 2>/dev/null || echo "parse_error")
        steps=$(python3 -c "import json; d=json.load(open('$json_file')); print(len(d.get('history', [])))" 2>/dev/null || echo "0")
        train_before=$(python3 -c "import json; d=json.load(open('$json_file')); print(f\"{d.get('train_before', 0):.6f}\" if d.get('train_before') else 'N/A')" 2>/dev/null || echo "N/A")
        train_after=$(python3 -c "import json; d=json.load(open('$json_file')); print(f\"{d.get('train_after', 0):.6f}\" if d.get('train_after') else 'N/A')" 2>/dev/null || echo "N/A")

        echo "  Status: $status"
        echo "  Steps: $steps / 160"
        echo "  Train loss: $train_before → $train_after"

        if [ -f "$BASE_DIR/$exp.log" ]; then
            last_log=$(tail -n 3 "$BASE_DIR/$exp.log" | grep -E "\[train\]|\[extract\]" | tail -1)
            if [ ! -z "$last_log" ]; then
                echo "  Last log: $last_log"
            fi
        fi
    else
        echo "  Status: not started or initializing"
    fi
    echo ""
done

echo "=========================================="
echo "PIDs:"
for pid_file in /tmp/scheduled_sampling_*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        name=$(basename "$pid_file" .pid | sed 's/scheduled_sampling_//')
        if ps -p $pid > /dev/null 2>&1; then
            echo "  $name (PID $pid): RUNNING"
        else
            echo "  $name (PID $pid): STOPPED"
        fi
    fi
done
echo "=========================================="
