# 面试现场 Demo 包

这个目录是会议时直接打开的材料。当前主视频使用 **RGB-consistent visual-stability protocol**，不是早期的 fixed-mix action grid。早期 fixed-mix 视频仍保留在 [`diagnostics/legacy_fixed_mix/`](diagnostics/legacy_fixed_mix/) 作为失败诊断，不能继续当作主视觉结果。

## 新增：保留动作较强旧版与长视频对照

- [动作响应与视觉稳定性的三列 A/D 对比](action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4)：原始 H3／旧 fixed-mix／RGB visual，完整保留漂移后段。
- [W/S/A/D 三列总览](action_vs_stability/original_action_stronger_visual_stable_grid_124.mp4)：同一首帧、prompt、seed 的四动作对照。
- [两套 checkpoint 的配置和限制](action_vs_stability/README.md)：旧版 A/D 符号较好，但分离仍远弱于原始 H3；视觉稳定版的 A 符号错误。
- [10 秒和 20 秒真实生成对照](long_horizon/README.md)：10 秒后段出现模糊和重影；20 秒 causal 严重退化，**长时视觉稳定性未通过**。两条都保留完整后段，也不作为 action preservation 的证明。


## 先说结论

H3-World 的 chunk-wise causal attention、persistent raw KV、clean commit 和 generated-history rollout 已经在真实 H3 checkpoint 上跑通，124 帧可以完整生成。早期主视频画面崩坏，主要是我在整理会议包时选用了没有 visual tail16 adapter、latent-only anchor protocol 的旧 fixed-mix action 实验；它能解码，不代表视觉质量合格。

换用后续的 RGB-consistent dual anchor（生成的 latent tail 先解码到 RGB，再经过 H3 image branch）和 tail16 visual QKV adapter 后，人物和车库结构在 124 帧末尾明显更完整。不过这次修复同时改变了 adapter、anchor、action routing 和 feedback 配置，不能把改善归因到单个组件；画面仍有模糊/ghosting，A/D action geometry 也没有恢复。

## 推荐播放顺序

1. [h3world_rgb_stable_action_grid_124_timed.mp4](annotated/h3world_rgb_stable_action_grid_124_timed.mp4)：四个动作的 4×2 并排总览；左侧原始 H3 30 steps，右侧 RGB-anchor causal prototype 8 steps/chunk，含单次 recorded end-to-end 时间。
2. [h3world_rgb_stable_W_original_vs_causal_timed.mp4](annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4)：一条完整 124-frame / 5.17 s 的单动作对比。
3. [h3world_rgb_stable_A_original_vs_causal_timed.mp4](annotated/h3world_rgb_stable_A_original_vs_causal_timed.mp4) 或 [h3world_rgb_stable_D_original_vs_causal_timed.mp4](annotated/h3world_rgb_stable_D_original_vs_causal_timed.mp4)：直接展示 A/D 的视觉稳定性和动作方向仍未完全恢复。
4. [stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4](visual_stability/stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4)：旧 latent-only anchor 与 RGB-consistent anchor 的修复证据。
5. [stage2_lite_rgb_endpoint_integrated_AD_39.mp4](stage2_lite/stage2_lite_rgb_endpoint_integrated_AD_39.mp4)：Stage2-lite student/critic/teacher 链路的可运行性证据。

如果只有 3 分钟，播放第 1 项，然后打开 [`METRICS.md`](METRICS.md) 和 [`FAIRNESS.md`](FAIRNESS.md)。

## 当前主方案配置

- 124 RGB frames / 5.17 s / 24 fps，5 latent frames/chunk，8 chunks；
- history window 5 chunks，persistent raw KV 放 CPU；
- 8 noisy denoiser steps/chunk，64 noisy forwards + 8 clean KV commits；
- `dynamic_last_frame_rgb_dual`，`global_retimed_rgb_prefix_last_image_dual_v2`；
- generated history，`action_prefix_mode=causal`，`action_feedback=true`，flow shift 2.22，seed 13；
- tail16 visual QKV adapter + action adapter；不包含官方 SolarWM Stage2 的完整 SGF/DMD 训练。

## 为什么早期视频会崩

会议包第一版把以下旧结果当成了正式主视频：

- `outputs/2026-10-03-02/final_fixed_mix124_8step/`；
- `causal_adapter=null`，没有 visual tail16 QKV adapter；
- `dynamic_last_frame_dual` + `global_retimed_latent_dual_v1` latent-only anchor；
- `action_prefix_mode=own`，`action_feedback=false`；
- 只有 fixed-mix action adapter。

这套协议在 generated-history 长 rollout 中会累积 temporal-VAE ghosting 和人物/车库 tearing。它的 MP4 可以播放，是编码/容器检查通过；但不应被当作视觉稳定性证明。后续 RGB visual run 使用了不同的视觉 adapter、anchor 和 action routing，因此它是“修复后的主视觉 demo”，不是对旧视频的单变量 ablation。

## 结果如何解释

当前 RGB-main 的单次记录为：原始 H3 30-step 约 441.5–454.2 s；causal RGB-anchor 约 673.4–767.9 s。causal 仍有 64 次 noisy forwards，所以不能宣称端到端加速。首块约 41.8–57.4 s，后续平均 chunk 约 75.1–88.6 s；GPU 峰值约 30.8–39.0 GiB，CPU raw KV 约 13.19 GiB。没有做 warmup 后重复均值，也没有把权重、KV 和 activation 峰值分别 instrument。

主视频的 A/D Farneback 水平光流为 A=`-0.784`、D=`-1.007`，A-D=`0.223`；原始 H3 的 A-D=`2.679`。A 的符号没有恢复，因此不能说四个方向完全保真。RGB anchor 解决的是视觉分解，generated-history 下的 action-conditioned score geometry 仍是未解决问题。

`RGB MAD` 和 chunk-boundary MAD 仅是相邻帧变化的描述性 proxy，较低值也可能来自模糊，不能直接当视频质量分数。没有 frame-aligned GT，所以本包不把 LPIPS/PSNR 当监督结果；FVD 需要足够多的真实/生成分布样本，VBench 也没有在本轮运行。

## 文件

- [`SLIDES.md`](SLIDES.md)：8 页展示提纲。
- [`MEETING_SCRIPT.md`](MEETING_SCRIPT.md)：约 5 分钟讲稿。
- [`METRICS.md`](METRICS.md) / [`METRICS.csv`](METRICS.csv)：当前 RGB-main 指标。
- [`FAIRNESS.md`](FAIRNESS.md)：输入公平性、步数和计时口径。
- [`source_metrics/rgb_visual/`](source_metrics/rgb_visual/)：主视频原始 JSON、flow 和 continuity 指标。
- [`diagnostics/legacy_fixed_mix/`](diagnostics/legacy_fixed_mix/)：旧主视频、旧指标和选片审计。
- [`annotated/`](annotated/)：带时间/步数标注的主 MP4。
