#!/bin/bash
# 重新运行 scheduled-sampling 实验，使用正确的学习率 3e-4

set -e

cd /home/qma/work/GWM/H3-World/code/causal

# 检查可用的 GPU
echo "Checking available GPUs..."
AVAILABLE_GPUS=($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk '$2 > 30000 {print $1}' | head -3))

if [ ${#AVAILABLE_GPUS[@]} -lt 3 ]; then
    echo "Error: Need at least 3 GPUs with >30GB free memory"
    echo "Available GPUs: ${AVAILABLE_GPUS[@]}"
    exit 1
fi

echo "Found ${#AVAILABLE_GPUS[@]} available GPUs: ${AVAILABLE_GPUS[@]}"
echo "Will use GPUs: ${AVAILABLE_GPUS[0]}, ${AVAILABLE_GPUS[1]}, ${AVAILABLE_GPUS[2]}"
echo ""

# 配置
OUTPUT_BASE="/home/qma/work/GWM/H3-World/outputs/2026-10-03-scheduled-sampling-lr3e4"
TEACHER_DIR="/home/qma/work/GWM/H3-World/outputs/2026-10-01-01/baseline30_seed13"

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# 1. Baseline (clean history only)
echo "=== Starting experiment 1/3: baseline_clean_only on GPU ${AVAILABLE_GPUS[0]} ==="
CUDA_VISIBLE_DEVICES=${AVAILABLE_GPUS[0]} python train_pretrained_multiaction.py \
  --teacher-dir "${TEACHER_DIR}" \
  --output-dir "${OUTPUT_BASE}/baseline_clean_only" \
  --lr 3e-4 \
  --steps 160 \
  --tail-blocks 4 \
  --target-chunks 0 1 2 3 4 5 6 7 \
  > /tmp/baseline_lr3e4.log 2>&1 &
PID1=$!
echo "Started PID $PID1"
echo ""

sleep 5

# 2. Fixed Mix 0.5
echo "=== Starting experiment 2/3: fixed_mix_0.5 on GPU ${AVAILABLE_GPUS[1]} ==="
CUDA_VISIBLE_DEVICES=${AVAILABLE_GPUS[1]} python train_pretrained_multiaction.py \
  --teacher-dir "${TEACHER_DIR}" \
  --output-dir "${OUTPUT_BASE}/fixed_mix_0.5" \
  --lr 3e-4 \
  --steps 160 \
  --tail-blocks 4 \
  --target-chunks 0 1 2 3 4 5 6 7 \
  --history-mix-schedule 0.5:0.5 \
  > /tmp/fixed_mix_lr3e4.log 2>&1 &
PID2=$!
echo "Started PID $PID2"
echo ""

sleep 5

# 3. Curriculum 0.0 → 0.5
echo "=== Starting experiment 3/3: curriculum_0.0_to_0.5 on GPU ${AVAILABLE_GPUS[2]} ==="
CUDA_VISIBLE_DEVICES=${AVAILABLE_GPUS[2]} python train_pretrained_multiaction.py \
  --teacher-dir "${TEACHER_DIR}" \
  --output-dir "${OUTPUT_BASE}/curriculum_0.0_to_0.5" \
  --lr 3e-4 \
  --steps 160 \
  --tail-blocks 4 \
  --target-chunks 0 1 2 3 4 5 6 7 \
  --history-mix-schedule 0.0:0.5 \
  > /tmp/curriculum_lr3e4.log 2>&1 &
PID3=$!
echo "Started PID $PID3"
echo ""

echo "All 3 experiments started!"
echo "PIDs: $PID1 $PID2 $PID3"
echo ""
echo "These experiments will take ~10-15 minutes each."
echo "Monitor progress with:"
echo "  tail -f /tmp/baseline_lr3e4.log"
echo "  tail -f /tmp/fixed_mix_lr3e4.log"
echo "  tail -f /tmp/curriculum_lr3e4.log"
