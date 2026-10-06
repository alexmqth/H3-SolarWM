# Legacy fixed-mix action adapter

`action_adapter.pt` 是原始实验 `H3-World/outputs/2026-10-02-scheduled-sampling/fixed_mix_0.5/action_adapter.pt` 的逐字节备份。它保留了比当前 RGB visual-main 更强的 A/D 方向响应，但 124 帧后段存在明显视觉漂移。

用于还原这次展示时，不加载 tail16 visual adapter：

```bash
python code/causal/benchmark.py --modes cached --num-frames 124 \
  --steps 8 --flow-shift 2.22 --chunk-frames 5 --history-chunks 5 \
  --action-preset A --seed 13 --cache-device cpu \
  --anchor-mode dynamic_last_frame_dual --action-prefix-mode own \
  --causal-action-adapter /path/to/submission/checkpoints/legacy_fixed_mix/action_adapter.pt \
  --out-dir outputs/legacy_fixed_mix_replay_A
```

`action_feedback` 保持关闭（不传 `--action-feedback`）。其 A/D signs 正确，但 A-D=0.4532 远低于原始 H3=2.6786。请把它作为动作响应与视觉稳定性的取舍展示，而不是 action control 已解决的结论。
