# Demo fairness and counting conventions

## Current main pair

当前会议主视频是 `annotated/h3world_rgb_stable_*`。每个左/右 pair 使用同一张 initial RGB image、同一 scene prompt、同一 action preset、同一 seed=13、同一初始 video/audio noise、同一分辨率和 124 帧长度。左侧是原始 H3-World 30-step full-horizon inference，右侧是 RGB-consistent causal prototype。

两条分支的 checkpoint/protocol 不同，这是比较“原始 H3 与 causal prototype”的设计，而不是同一个 checkpoint 的重放。右侧固定为：tail16 visual QKV adapter、action adapter、RGB dual anchor、generated history、causal action prefix、action feedback、CPU raw KV。早期 fixed-mix grid 使用了 latent-only anchor 和另一套 adapter，已移动到 `diagnostics/legacy_fixed_mix/`，不混入本表。

## Step counting

- Original H3：30 次完整 horizon denoiser evaluations。
- Causal：8 steps/chunk × 8 chunks = 64 次 noisy denoiser forwards，另有 8 次 clean KV commits。
- `8 steps/chunk` 只表示每个 chunk 的 solver steps，不能说成“8 次 forward 生成完整 124 帧”。
- clean commit 是把 clean latent 的 raw K/V 写入 history 的额外 forward，单独计数。

## Timing and memory

主表是每个 action 一次已完成的 recorded run，不是 warmup 后重复均值。端到端时间包含 shared setup、conditioning、sampling、RGB anchor decode/re-encode、VAE/CPU 转移和 MP4 写入；sampling-only 也保存在 `source_metrics/rgb_visual/summary.json`。GPU 数值是 `torch.cuda.max_memory_allocated` 的整次运行峰值；CPU KV 是 raw-KV cache 峰值。没有把权重、KV 和 activation 的显存分别拆开，因此不能从总峰值推导组件占用。

首块时间是内部首个 causal chunk 的生成/提交时间，不是视频播放器实际显示第一帧的 streaming latency；当前 benchmark 在末尾统一完成解码/编码，不能把它宣传成端到端交互延迟。

## Quality and continuity

本轮没有 frame-aligned ground-truth future。表中使用：

- mean/p95 adjacent-frame RGB MAD；
- nominal chunk-boundary frames 17、34、51、68、85、102、119 的 RGB MAD；
- signed horizontal Farneback flow；
- 完整解码、contact sheet 和人物/车库结构的人工检查。

MAD 是运动/连续性 proxy，不是质量分数；Farneback flow 是 image-space motion proxy，不是 action accuracy。FVD 需要真实和生成视频分布以及足够样本，LPIPS/PSNR 需要 paired reference，VBench 需要明确的评测协议；这些本轮都没有作为主结果运行。
