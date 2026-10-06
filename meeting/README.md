# 面试现场 Demo 包

这个目录只保留开会时需要展示的材料。推荐先播放带时间标注的 124-frame W/S/A/D 总览，再播放一条 W 或 A/D 并排视频，最后用指标表解释为什么这是一个 causal feasibility prototype，而不是已经完成的 action-preserving Stage2 模型。

## 30 秒结论

> 我把 H3-World 改成了 SolarWM 风格的 chunk-wise causal rollout：5 latent frames/chunk、persistent raw KV、clean commit、generated history 和 8 steps/chunk。124 帧视频可以完整生成，RGB-consistent anchor 也修复了后段人物分解。当前瓶颈不是能不能 causalize，而是 generated-history 改变了 H3 原始 action-conditioned score geometry：A/D 仍有响应，但方向幅度明显变弱，严格 action gate 尚未通过。

## 推荐播放顺序

1. [h3world_final_action_grid_124_timed.mp4](annotated/h3world_final_action_grid_124_timed.mp4)：四个动作、左侧原始 H3 30 steps、右侧 causal Stage1 8 steps/chunk；每个 panel 已标注 recorded end-to-end 时间。
2. [h3world_final_W_original_vs_causal_timed.mp4](annotated/h3world_final_W_original_vs_causal_timed.mp4)：展示一条完整 5.17 秒并排视频，说明人物和车库结构仍可保持。
3. [stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4](visual_stability/stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4)：说明旧 latent-only anchor 的人物分解如何被 RGB-consistent anchor 修复。
4. [stage2_rgb_endpoint_visual_stable_AD_124.mp4](visual_stability/stage2_rgb_endpoint_visual_stable_AD_124.mp4)：说明视觉稳定性已经改善，但不要把它说成 A/D 方向恢复。
5. [stage2_lite_rgb_endpoint_integrated_AD_39.mp4](stage2_lite/stage2_lite_rgb_endpoint_integrated_AD_39.mp4)：说明 Stage2-lite 的 student/critic/teacher 链路可以运行。

如果只有 3 分钟，播放第 1 项，然后直接打开 [METRICS.md](METRICS.md) 和 [SLIDES.md](SLIDES.md)。

## 主 demo 的公平性

正式 124-frame grid 使用 fixed-mix action adapter 和 dynamic latent dual-anchor protocol；它使用同一张 initial RGB image、同一 scene prompt、同一 action sequence、同一 seed=13、同一 initial video/audio noise、同一分辨率和同一帧数。四个动作 W/S/A/D 分别固定到 8 个 causal chunks。左侧是原始 H3 30-step full-horizon inference；右侧是 causal prototype 的 8 steps/chunk。

步数口径必须说清楚：

- 原始 H3：30 次完整 horizon denoiser forwards；
- causal：8 steps/chunk x 8 chunks = 64 次 noisy denoiser forwards，外加 8 次 clean KV commit；
- 因此 causal 的“8 steps”不是整个 124 帧只调用 8 次网络。

视频上的耗时是各自实验日志中的单次 recorded end-to-end wall time，包含 shared setup、conditioning、sampling 和 VAE/编码流程。它还不是 warmup 后多次均值；表格已明确标记这一点。正式报告不能把当前 causal prototype 宣称成端到端加速版，因为当前主 demo 的 forward 数量实际上高于 30-step baseline。

## 结果怎么解释

| 要展示的事实 | 证据 |
|---|---|
| causal chunk/KV 链路真实运行 | 124 帧、8 chunks、64 noisy forwards、8 clean commits、replay error=0 |
| 视觉稳定性改善 | RGB anchor 对比、124 帧 W/A/D 人物和停车场结构保持 |
| action 仍有差异但没有完整保真 | 原始 A-D flow=2.679，formal causal fixed-mix A-D=0.453 |
| generated-history action geometry 仍是问题 | 39 帧严格 gate `flow(A)>0, flow(D)<0, A-D>1.0` 未通过；frozen causal/teacher delta cosine 约 -0.015 |
| Stage2-lite 不是官方 Stage2 | shared backbone + small critic adapter 可运行，但 integrated A/D gate 仍失败 |

## 文件

- [SLIDES.md](SLIDES.md)：8 页展示提纲。
- [MEETING_SCRIPT.md](MEETING_SCRIPT.md)：可以直接照着讲的 5 分钟讲稿。
- [METRICS.md](METRICS.md)：人类可读指标表。
- [METRICS.csv](METRICS.csv)：逐动作、逐方法的机器可读指标。
- [FAIRNESS.md](FAIRNESS.md)：公平性和步数口径。
- [annotated/](annotated/)：增加 recorded end-to-end 时间标注的 MP4。
- [final/](final/)：原始未二次标注的正式对比视频。
- [source_metrics/](source_metrics/)：生成指标所依据的 JSON/报告原件。

## 现场回答“是否更快”

建议回答：

> 当前 prototype 的重点是 causal execution 和历史计算复用，不是已经得到端到端 wall-clock speedup。正式 grid 中原始 30-step 是约 441–454 秒，causal 8-step/chunk 是约 383–452 秒，而且 causal 需要 64 noisy forwards。KV cache 的价值在于后续 chunk 不重复计算全部历史，首块可以独立提交并支持流式接口；若要让总 wall time 真正下降，还需要减少每 chunk 的 solver evaluations，并完成 Stage2 的少步蒸馏。

## 现场回答“动作保留了吗”

建议回答：

> formal fixed-mix 124-frame grid 仍能看到 action-dependent difference，A 的 flow 为正、D 的 flow 为负，但 A-D 从 teacher 的 2.679 降到 0.453。更严格的 RGB generated-history gate 没有通过，所以我把结论写成 causal feasibility + visual stability recovery，而不是 full action preservation。剩余问题是 action-conditioned score geometry 和 rollout distribution shift。
