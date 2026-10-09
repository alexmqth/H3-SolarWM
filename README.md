# H3-World × SolarWM: causalization interview submission

2026-10-09 16:16：E2两臂各4更新、六组局部视频评测及48次held-out诊断全部完成。停车场两份历史的A/D符号均保留，但A分支重影仍在，FM+action没有一致优于FM-only；局部动作＋结构联合gate仍为No-Go。本轮不自动扩训，不进入AnyFlow/Stage2。 [完整结果与视频](reports/stage1_anyflow/real_transition_windows/FINAL_RESULTS.md)。

这是面试题的可审阅提交包：在 H3-World 中验证 SolarWM 风格的 causal chunk、KV cache 和少步生成思路，同时检查 H3-World 原有 action control 是否保留。

包内包含 causal/KV 实现、最小 Stage2-lite 训练链路、实验 adapter、可播放对比视频和报告；不包含 MiniMax-H3 33B 基础权重、H3-World 原始 LoRA、数据集、缓存、latent/conditioning 中间文件。

## 展示入口

- [动作控制与视觉稳定性的三列 A/D 对照](meeting/action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4)
- [W/S/A/D 三列总览：原始／旧版／新版](meeting/action_vs_stability/original_action_stronger_visual_stable_grid_124.mp4)
- [10/20 秒原始 H3 与 RGB causal 的长视频对照](meeting/long_horizon/README.md)
- [会议播放说明与完整指标](meeting/README.md)

旧 fixed-mix 的视频、adapter 和配置与当前稳定版一起保留。旧版 action response 相对更强（A 正、D 负），但后段视觉漂移；稳定版结构更完整但 A/D 控制弱。两种限制都在对比视频中直接展示。

**最新长度测试：20 秒 causal 视觉稳定性失败。** 同一 RGB checkpoint 在 10 秒末段已有模糊和重影；20 秒样本约 10 秒开始严重雾化，15 秒后人物和场景难以辨认。这里的“稳定版”仅指 124 帧上相对旧版的视觉改善，不是长视频稳定性保证。完整失败后段保留在视频中。


## 当前结论

- **04时段更正：** 官方H3 Stage1冻结目标时间MLP；旧pilot额外训练了它。[参数策略复核与修正](STAGE1_PARAMETER_POLICY.md)记录了证据；真实33B冻结分支训练及评测已完成，单独冻结没有解决画质和动作问题。
- [Stage1 TF-AnyFlow实现](STAGE1_ANYFLOW.md)与[验收清单](STAGE1_ACCEPTANCE.md)：时间条件、有限差分loss、有限区间采样、native-FP32和完整历史梯度均已接通，公式/梯度/恢复检查通过。既定136对照及完整4/8步视频仍未通过画质/action gate。真实ABot FM48与后续密度对照均已收尾并未通过，当前按[三实验协议](reports/stage1_anyflow/three_experiments/PROTOCOL.md)优先校准Original局部动作信息流。工程检查通过不等于视觉验收；会议视频仍来自旧FM checkpoint。
- H3-World 已跑通 chunk-wise causal attention、persistent raw KV、clean KV commit 和 generated-history rollout；124 帧（约 5.17 秒）可完整解码。
- 早期 fixed-mix 主视频出现人物透明、车库 tearing，是旧 anchor protocol 和缺少 visual adapter 的 generated-history 漂移；它后来被错误地选成会议主展示。旧视频已移到 `meeting/diagnostics/legacy_fixed_mix/`。
- 当前 124 帧会议主视频使用 RGB-consistent dual anchor（generated latent tail 解码到 RGB，再经过 H3 image branch）和 tail16 visual QKV adapter。该长度下人物和场景结构比旧主视频更完整，但仍有 blur/ghosting；扩展到 481 帧后视觉崩坏。
- SolarWM-inspired Stage2-lite（student self-rollout、fake-score critic、frozen teacher、DMD surrogate）可以运行在 shared H3 backbone 上，但没有恢复严格的 A/D action gate。
- 当前 RGB-main 124 帧的 A/D flow 为 A=`-0.784`、D=`-1.007`、A-D=`0.223`；原始 H3 为 A=`+1.077`、D=`-1.602`、A-D=`2.679`。因此不声称完整保留四方向 action control，也不声称端到端加速。

可 defensibly 写成：

> We successfully causalized H3-World with chunk-wise attention and persistent KV caching. RGB-consistent anchoring and visual adaptation improve visual coherence at 124 frames relative to the earlier fixed-mix prototype, but the same checkpoint suffers severe visual collapse in the 20-second rollout. The strict A/D action gate is also not reached. The prototype demonstrates causal execution with bounded history KV and exposes unresolved action-control and long-horizon generated-history failures; it does not establish long-video stability or full preservation of H3-World action control.


## 文档和目录

- `INTERVIEW_ANSWER.md`：面试题逐项回答。
- `REPORT.md`：**简短实验报告**（方法说明、最小原型与可行性、Demo 与指标、长视频结果、结论）。
- `EXPERIMENT_LOG.md`：精简的实验时间线。
- `EXPERIMENT_REPORT.md`：39/124 帧协议、视觉修复、Stage2-lite、action geometry 和 243/481 帧长度测试。
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

`meeting/diagnostics/legacy_fixed_mix/` 中旧的 `h3world_final_*` 文件仍作为历史 fixed-mix 证据保留；它们不再是会议主入口。所有主视频均已用 PyAV 检查为 H.264/YUV420P、24 fps、完整帧数。

## 重要限制

运行代码仍需外部 MiniMax-H3 FL2VA 权重、H3-World step-10000 LoRA 和 patched DiffSynth checkout；见 `REPRODUCE.md`。当前 RGB-main 的耗时是单次记录，且 64 noisy forwards/chunk rollout 多于原始 30 full-horizon forwards，不应宣传为总 wall-clock speedup。
