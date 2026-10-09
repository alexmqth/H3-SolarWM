# H3-World × SolarWM：在 H3-World 中验证因果少步生成

面试题：把 SolarWM 的因果分块、KV Cache 和少步生成思路迁移到 H3-World，验证能否改善长视频生成效率，并检查 H3-World 原有的动作控制是否保留。

## 一句话结论

**因果分块 + 持久 KV 在真实 33B H3 上工程可行；但到目前为止，没有一个版本同时做到"动作对、画面稳、更快"。** 动作失配在因果化本身就出现（早于 AnyFlow 和 Stage2），后续在伪标签和真实录制视频上的训练都没有修复它。研究已冻结。

> We causalized H3-World with chunk-wise attention and persistent KV caching. RGB-consistent anchoring improves visual coherence at 124 frames, but the same checkpoint collapses in a 20-second rollout, and the strict A/D action gate is never reached. The prototype shows causal execution with bounded history KV; it does not establish speedup, long-video stability, or preserved action control.

## 从这里开始看

| 想了解 | 打开 |
|---|---|
| 面试交付（简短报告） | [REPORT.md](REPORT.md) |
| 面试题逐项回答 | [INTERVIEW_ANSWER.md](INTERVIEW_ANSWER.md) |
| 现场展示（主视频、讲稿、指标） | [meeting/README.md](meeting/README.md) |
| 每个实验做了什么、结果如何 | 下面的**实验地图** |
| 因果化单独做到哪一步、哪些版本已保存 | [因果基线与视频](docs/CAUSAL_BASELINE.md) |
| 怎么复现 | [REPRODUCE.md](REPRODUCE.md) |

## 实验地图

结论栏：**通过** = 达到该实验的目的；**部分** = 有改善但有明显副作用；**未通过** = 没达到验收；**定位** = 诊断类实验，回答"问题出在哪"，本身不修复。动作指标 A−D 是 A、D 两个动作的水平光流之差，原始 H3 为 2.0～2.7，门槛 > 1.0。

### 阶段一：把 H3-World 改成因果（会议主视频来自这一阶段）

| # | 实验 | 目的 | 做法 | 结果 | 结论 | 证据 |
|---|---|---|---|---|---|---|
| 1 | 因果分块 + 持久 KV | H3 能否按块生成并复用历史 K/V | 5 个 latent 一块、5 块滑窗；块去噪完后用干净 latent 写入每层 K/V，放 CPU | 124 帧跑通，replay 误差 0；64 次前向对原始 30 次，没有加速 | 通过（工程） | [01](experiments/01_causal_chunk_kv_rollout/README.md) |
| 2 | 旧 fixed-mix 动作适配 | 自生成历史下保住动作 | 动作 residual + scheduled sampling，latent anchor，无视觉适配器 | A−D 0.45、符号对；约 4 秒后背景撕裂，A/D 比 W 严重 | 部分 | [旧版视频与指标](meeting/diagnostics/legacy_fixed_mix/) |
| 3 | RGB anchor 视觉修复 | 修人物分解 | 末帧解码成 RGB 再走 H3 图像分支 + tail16 视觉 QKV 适配 | 124 帧结构完整；A−D 0.22，A 方向错 | 部分（**会议主片**） | [02](experiments/02_rgb_anchor_visual_drift_repair/README.md) |
| 4 | Stage2-lite | 跑通 SolarWM Stage2 的角色 | 共享骨干轮换 student / critic / teacher，DMD surrogate | 链路可运行；A−D 0.31 | 未通过 | [03](experiments/03_stage2_lite_critic_dmd/README.md) |
| 5 | 动作几何审计 | 动作为什么失效 | 同一生成状态上比较因果与原始的 A/D 差异向量；8 条修复路线 | 幅度比 0.88，**方向余弦 −0.015**（几乎正交）；8 条路线均未过 | 定位 | [04](experiments/04_action_geometry_audit/README.md) · [05](experiments/05_corrected_own_history_endpoint/README.md) |
| 6 | 10 / 20 秒长视频 | 长时是否稳定 | 同一 RGB checkpoint 生成 243 / 481 帧 | 10 秒后段模糊；**20 秒约 10 秒起严重雾化**；早先"原始 H3 长片 OOM"已撤回（mask 实现问题） | 未通过 | [06](experiments/06_long_rollout_and_tradeoff/README.md) · [20 秒视频](meeting/long_horizon/original_vs_rgb_visual_W_20s_481f.mp4) |

### 阶段二：Stage1 / AnyFlow 与真实视频训练（结果不在会议主片里）

| # | 实验 | 目的 | 数据 / 做法 | 结果 | 结论 | 证据 |
|---|---|---|---|---|---|---|
| 7 | AnyFlow 训练 | 学会少步（4/8 步） | **Original H3 生成的停车场 A/D 伪标签**；TF-AnyFlow 128 步，另做 136 步有限预算对照 | 8 步 A−D 0.76 但画面漂移；4 步严重雾化；136 步为 0.58 / 0.64 | 未通过 | [128 步](reports/stage1_anyflow/05_runtime/parallel_resume68_to128/STEP128_RESULTS.md) · [136 步对照](reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/FINAL_RESULTS.md) |
| 8 | AnyFlow 同状态诊断 | 排除实现错误 | 固定历史只换当前动作，查对齐、路由、KV | 未发现错位、泄漏或 KV 污染；方向余弦平均 0.038 | 定位 | [07](experiments/07_anyflow_fixed_state_action_diagnostic/README.md) |
| 9 | AnyFlow 之前就失配 | 问题在训练前还是训练中 | 原始权重只改成因果，不训练 | 整体速度余弦 0.996，**动作差分余弦 0.06** → 因果化本身改变了动作几何 | 定位 | [08](experiments/08_causal_field_before_anyflow/README.md) |
| 10 | 真实视频 FM48 | 用真实数据训练能否修复 | **ABot 录制的游戏视频**（16 训练 / 8 验证）；普通 FM，48 更新 | 动作差分余弦 0.0175→0.0176；停车场 A−D −0.02；自然场景生成历史约 23 帧后人物分解 | 未通过 | [完整验收](reports/stage1_anyflow/01_real_video/real_abot_fm/FM48_COMPLETE_REVIEW.md) · [生成历史对比视频](reports/stage1_anyflow/01_real_video/real_abot_fm/report/trained_complete_generated30/dfec8ed3237860eba14d67c089ecd041_D_1750_comparison.mp4) |
| 11 | 噪声采样分布对照 | 换 sigma 采样能否帮助 | 同数据同划分，只改训练 sigma 分布，48 更新 | 停车场 A−D −0.03（30 步）/ 0.07（8 步） | 未通过 | [结果](reports/stage1_anyflow/01_real_video/fm_density_control/FINAL_RESULTS.md) |
| 12 | Original 条件校准 | 找回可信的正控 | 无训练；恢复 H3 原生 text 时间和单首帧条件 | A−D 恢复到 2.23 | 通过（正控） | [09](experiments/09_original_conditioning_calibration/README.md) |
| 13 | 历史条件协议 | 历史怎么喂会影响动作吗 | 历史改用与当前 sigma 一致的加噪版本 | 修正了当前 D 方向，但出现多重手臂重影 | 部分 | [10](experiments/10_history_conditioning_action_response/README.md) |
| 14 | E2 动作后果监督 | 加动作监督是否比普通 FM 好 | 真实 ABot 过渡窗口；FM-only 与 FM+action 各 4 更新，同初始化同数据 | 动作项无一致收益，A 分支重影仍在 → 冻结，不扩训 | 未通过 | [结果与 6 条视频](reports/stage1_anyflow/01_real_video/real_transition_windows/FINAL_RESULTS.md) |
| 15 | Stage2 工程准备 | 为完整 Stage2 备好原语 | 小模型上验证 DMD 方向、FMBS、共享角色隔离 | 工程检查通过；不是 33B 效果 | 通过（工程） | [说明](reports/stage1_anyflow/06_stage2_preparation/README.md) |
| 16 | 干净环境验收 | 提交包能否独立复现 | 新 venv + 新源码；15 项 KV/因果测试、训练 smoke、33B 39 帧推理 | 全部通过；画质与动作结论不变 | 通过（工程） | [验收](reports/final_acceptance/README.md) |

**实验之间的关系：** 1→3 解决"能跑"和"画面稳"，但 2↔3 暴露动作与画面的取舍（[三列对比视频](meeting/action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4)）；5、8、9 把动作失配定位到"因果化本身"；7、10、11、14 说明在伪标签或真实视频上继续训练都没修好；12、13 是目前仅有的正向线索。

## 目录结构

```text
README.md            本页：结论与实验地图
REPORT.md            简短实验报告（面试交付）
INTERVIEW_ANSWER.md  面试题逐项回答
REPRODUCE.md         环境、补丁、外部权重与运行命令
experiments/         每个关键实验一页：问题、做法、证据、结论（编号 01–10 对应上表）
meeting/             现场展示：主视频、讲稿、指标、公平性说明、长视频和取舍对比
reports/             详细报告与原始证据（指标 JSON、视频、帧图）
  stage1_anyflow/      阶段二的全部实验，按 7 类归档，见其 README
docs/                长文档：完整实验报告、精简时间线、局限、Stage1 设计与验收
  archive/             历史进度流水账、旧计划、旧文件清单（只读参考）
code/                因果 / KV / 训练代码与 DiffSynth 补丁
checkpoints/         小型实验 adapter（不含 33B 权重）
scripts/ tests/      运行时准备、推理验证与单元测试
```

## 外部依赖

运行代码需要 MiniMax-H3 FL2VA 基础权重、H3-World step-10000 LoRA 和打过补丁的 DiffSynth，均不在仓库内，见 [REPRODUCE.md](REPRODUCE.md)。所有耗时都是共享硬件上的单次记录，不是 warmup 后的多次均值。
