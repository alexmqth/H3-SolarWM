#!/bin/bash
# Summarize all scheduled-sampling and Task 1 results
# Usage: bash summarize_results.sh

echo "=========================================="
echo "GWM Scheduled-Sampling Results Summary"
echo "Generated: $(date)"
echo "=========================================="
echo ""

# Check Task 1 (124-frame baseline)
echo "=== Task 1: stage1_action_generated_hidden124 (124 frames) ==="
TASK1_JSON="/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/2026-10-02-10/stage1_action_generated_hidden124/training.json"
if [ -f "$TASK1_JSON" ]; then
    python3 << EOF
import json
with open('$TASK1_JSON') as f:
    d = json.load(f)
    print(f"  Status: {d.get('status')}")
    print(f"  Steps: {len(d.get('history', []))}")
    if d.get('train_after'):
        print(f"  Train loss: {d['train_before']:.6f} → {d['train_after']:.6f}")
        print(f"  Val loss: {d['validation_before']:.6f} → {d['validation_after']:.6f}")
        print(f"  Time: {d.get('wall_seconds')} s")
        print(f"  Peak mem: {d.get('allocated_peak_MiB')} MiB")
    else:
        print("  Training: in progress or finalizing")
EOF
else
    echo "  Not found"
fi
echo ""

# Check Task 2 experiments (39-frame scheduled-sampling)
echo "=== Task 2: Scheduled-Sampling Experiments (39 frames) ==="
BASE_DIR="/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/2026-10-02-scheduled-sampling"

for exp in baseline_clean_only fixed_mix_0.5 curriculum_0.0_to_0.5; do
    echo ""
    echo "--- $exp ---"
    json_file="$BASE_DIR/$exp/training.json"

    if [ -f "$json_file" ]; then
        python3 << EOF
import json
with open('$json_file') as f:
    d = json.load(f)
    print(f"  Status: {d.get('status')}")
    print(f"  Steps: {len(d.get('history', []))}/160")
    if d.get('train_after'):
        print(f"  Train loss: {d['train_before']:.6f} → {d['train_after']:.6f}")
        print(f"  Val loss: {d['validation_before']:.6f} → {d['validation_after']:.6f}")
        print(f"  Time: {d.get('wall_seconds')} s")
        print(f"  Peak mem: {d.get('allocated_peak_MiB')} MiB")
    else:
        print("  Training: in progress")
EOF
    else
        echo "  Not started or initializing"
    fi

    # Check if benchmark results exist
    benchmark_flow="$BASE_DIR/${exp}_benchmark/flow.json"
    if [ -f "$benchmark_flow" ]; then
        echo "  Benchmark results:"
        python3 -c "import json; d=json.load(open('$benchmark_flow')); print(f\"    A-D horizontal flow: {d.get('A_minus_D_horizontal', 'N/A')}\")"
    fi
done

echo ""
echo "=========================================="
echo "Expected Results:"
echo "=========================================="
echo ""
echo "Baseline (clean only):"
echo "  • A-D flow ≈ +0.956 (known upper bound from clean-history)"
echo ""
echo "Fixed mix 0.5:"
echo "  • If effective: A-D flow > 0.015 (baseline generated-history)"
echo "  • Test whether fixed mixing helps"
echo ""
echo "Curriculum 0.0→0.5:"
echo "  • If effective: A-D flow ≥ fixed_mix"
echo "  • Should be best of the three if curriculum learning works"
echo ""
echo "Success criterion:"
echo "  • Curriculum achieves A-D flow > 0.1 on generated-history rollout"
echo "  • If yes → extend to 124 frames"
echo "  • If no → investigate why scheduled-sampling fails"
echo ""
