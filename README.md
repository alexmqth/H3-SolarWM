# H3-World × SolarWM: causalization interview submission

这是面试题的可审阅提交包：在 H3-World 中验证 SolarWM 风格的 causal chunk、KV cache 和少步生成思路，同时检查 H3-World 原有 action control 是否保留。

包内包含 causal/KV 实现、最小 Stage2-lite 训练链路、实验 adapter、可播放对比视频和报告；不包含 MiniMax-H3 33B 基础权重、H3-World 原始 LoRA、数据集、缓存、latent/conditioning 中间文件。

## 当前结论

- H3-World 已跑通 chunk-wise causal attention、persistent raw KV、clean KV commit 和 generated-history rollout；124 帧（约 5.17 秒）可完整解码。
- 早期 fixed-mix 主视频出现人物透明、车库 tearing，是旧 anchor protocol 和缺少 visual adapter 的 generated-history 漂移；它后来被错误地选成会议主展示。旧视频已移到 `meeting/diagnostics/legacy_fixed_mix/`。
- 当前会议主视频使用 RGB-consistent dual anchor（generated latent tail 解码到 RGB，再经过 H3 image branch）和 tail16 visual QKV adapter。人物和场景结构比旧主视频稳定得多，但仍有 blur/ghosting。
- SolarWM-inspired Stage2-lite（student self-rollout、fake-score critic、frozen teacher、DMD surrogate）可以运行在 shared H3 backbone 上，但没有恢复严格的 A/D action gate。
- 当前 RGB-main 124 帧的 A/D flow 为 A=`-0.784`、D=`-1.007`、A-D=`0.223`；原始 H3 为 A=`+1.077`、D=`-1.602`、A-D=`2.679`。因此不声称完整保留四方向 action control，也不声称端到端加速。

可 defensibly 写成：

> We successfully causalized H3-World with chunk-wise attention and persistent KV caching. RGB-consistent anchoring and visual adaptation recover much of the long-horizon visual stability. However, generated-history causal rollout changes the action-conditioned score geometry, and the strict A/D action gate is not reached. The prototype demonstrates causal feasibility and isolates the remaining action-pathway/rollout-distribution problem; it does not claim full preservation of H3-World action control.

## 文档和目录

- `INTERVIEW_ANSWER.md`：面试题逐项回答。
- `EXPERIMENT_REPORT.md`：39/124 帧协议、视觉修复、Stage2-lite 和 action geometry 结果。
- `REPRODUCE.md`：环境、patch、外部权重和运行命令。
- `LIMITATIONS.md`：不应过度声称的结论和后续实验。
- `NEXT_PLAN.md`：action geometry 后续计划和负结果记录。
- `meeting/`：会议主视频、指标、slides、讲稿和选片审计。
- `breakthrough/`：按突破整理的问题、方案、证据视频和限制。
- `code/`：causal、action routing、训练和 DiffSynth patches。
- `checkpoints/`：小型 experiment adapters，不含 33B model。

## 会议主视频

主入口已经从旧 fixed-mix 改为：

| 视频 | 含义 |
|---|---|
| [`meeting/annotated/h3world_rgb_stable_action_grid_124_timed.mp4`](meeting/annotated/h3world_rgb_stable_action_grid_124_timed.mp4) | 当前 W/S/A/D 四动作 124 帧并排总览 |
| [`meeting/annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4`](meeting/annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4) | W：原始 30-step 对 RGB-main causal |
| [`meeting/annotated/h3world_rgb_stable_A_original_vs_causal_timed.mp4`](meeting/annotated/h3world_rgb_stable_A_original_vs_causal_timed.mp4) | A：原始 30-step 对 RGB-main causal |
| [`meeting/annotated/h3world_rgb_stable_D_original_vs_causal_timed.mp4`](meeting/annotated/h3world_rgb_stable_D_original_vs_causal_timed.mp4) | D：原始 30-step 对 RGB-main causal |
| [`meeting/annotated/h3world_rgb_stable_S_original_vs_causal_timed.mp4`](meeting/annotated/h3world_rgb_stable_S_original_vs_causal_timed.mp4) | S：原始 30-step 对 RGB-main causal |
| [`meeting/visual_stability/stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4`](meeting/visual_stability/stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4) | RGB anchor 视觉修复诊断 |
| [`meeting/stage2_lite/stage2_lite_rgb_endpoint_integrated_AD_39.mp4`](meeting/stage2_lite/stage2_lite_rgb_endpoint_integrated_AD_39.mp4) | Stage2-lite 链路诊断 |

`videos/final/` 中旧的 `h3world_final_*` 文件仍作为历史 fixed-mix 证据保留；它们不再是会议主入口。所有主视频均已用 PyAV 检查为 H.264/YUV420P、24 fps、完整帧数。

## 重要限制

运行代码仍需外部 MiniMax-H3 FL2VA 权重、H3-World step-10000 LoRA 和 patched DiffSynth checkout；见 `REPRODUCE.md`。当前 RGB-main 的耗时是单次记录，且 64 noisy forwards/chunk rollout 多于原始 30 full-horizon forwards，不应宣传为总 wall-clock speedup。
