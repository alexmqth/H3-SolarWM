# CVPR 研究问题的现有证据图谱（Worker，2026-10-16）

**目标问题：**把已学会语言动作的双向 H3-World 改为严格因果、可复用历史状态的生成器时，怎样保留当前动作对未来画面的影响？本页只整理已有文件；没有新模型调用、训练或文献优先权结论。`速度差分改变`、`视频出现不同`、`语义动作正确`分别是不同层级的证据。

| 观察与来源 | 可确认的事实 | 对 H1/H2 的意义与主要混杂 |
|---|---|---|
| [Original routing 源码审计](../../reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/README.md) | Original Single-Egress 中 `V_i` 只直接读 `A_i`，`A_i` 只直接读 `V_i`；公共 prefix 可读 video。旧 `action_prefix_mode=causal` 让当前 video 直接读过去 action，而旧基础缓存还关闭公共 prefix→video。缓存本身只存历史 video K/V/RoPE，不存 action tokens。 | **H1 的结构前提成立**：几条合法/新增边确实不同。结构差异不等于动作失控的实证原因。过去动作仍可能经过去 video 表示间接传播。 |
| [固定状态路由/KV审计](../../reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md) | 同一停车场两份 history×三 sigma 的 R1→R1-P（关闭公共反馈）、R1-P→R2（仅视频三角化）、R2→R3（sigma0 历史提交冻结）、R3→R4（扩过去动作直读）均改变 A/D velocity delta。R1-P/R2 的 delta cosine 均值约 0.1175；R2/R3 约 0.6400。匹配图、时间、受控 GEMM/SDPA 后，重算与缓存的历史 KV/velocity 在6状态逐元素相同。 | **支持做 H1 干预，但未证明视频语义**；一个场景/seed/插值状态，非真实 ABot 行为。R2 已在无持久 KV 时失配，不能把所有问题归因缓存。R2/R3 差别是提交时间/条件冻结，不能表述为同图缓存执行错误；默认数值后端曾产生非零误差，必须复用数值校准。各 cosine 不可相加为因果贡献百分比。 |
| [V1 早期](../../reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/RESULTS.md) 与 [V3 EXP-002](../../experiments/EXP-002_native_cached/judge/FINAL_REVIEW.md) | 旧 RGB-dual/5-latent 路由候选首块 delta 身份恢复、后块没有恢复；V3 原生 Single I0、12→5、own/current-prefix、30NFE、严格因果 raw KV 在停车场共用首12下，第二块 AA/AD flow 为 +1.347/−1.458，人物与场景可辨。 | 因果/KV **并非普遍抹掉动作**。两个版本同时改了 I0、time、历史、partition、prefix 等，不能用 V1→V3 差异给某条边单独归因。停车场单 scene/seed 不能代表 ABot 跨场景。 |
| [V3 后段切换 EXP-010](../../experiments/EXP-010_v3_fm8_sw158/judge/FINAL_REVIEW.md) | 同一 AA124 历史后，C7 A继续/D切换的水平 flow 为 +0.439/+0.034；C8 各自历史下 +0.587/−0.858。D 首切换块弱，下一块方向更清楚。 | **H2 的现象线索**，不是证据：运动惯性、动作实现、噪声、history 分叉、8NFE 都可能解释延迟；C8 不再是同历史对照。 |
| [ABot 多初图同历史分叉 EXP-014](../../experiments/EXP-014_v3_multiscene_teacher/README.md) | V3 FM30 在 `s0_43866101` 同一自生成 C1 下 AA/AD 新17帧明显分叉，中央 flow +42.147/−27.459；`s1` D反向较清楚而 A 弱；`s2` D 未清楚反转；`s3_b784d995` 两路几乎同向（+66.178/+58.664），人物仍可辨。 | 严格因果模型在 ABot 初图上的动作响应**有场景差异**，适合作为受控 pilot 选择依据。A/D 为合成控制，不是这些录屏的真实 GT；中央 flow 主要可能包含相机运动，不能单独给角色方向打标签。当前没有相同状态下 Original-visible 三路拓扑比较。 |
| [真实 ABot GT-history EXP-019](../../experiments/EXP-019_v3_mixed_action_validation/judge/FINAL_REVIEW.md)、[Original EXP-020](../../experiments/EXP-020_original_mixed_action_reference/README.md) | 四场景 GT C1→C2 的 FM30/FM8 局部视觉可用、混合动作语义 PARTIAL；Original `b784d995` 从 I0 联合生成全56帧，动作切换也缺可信反事实。 | 不能把“Original vs causal”视频当拓扑单因素：输入历史、生成范围和采样不同。弱响应可能是任务/场景本身困难，不先验归因 H1/H2。 |
| [真实两片 FM EXP-027](../../experiments/EXP-027_abot_baseline_diagnosis/judge/FINAL_REVIEW.md) | 两个不同 I0/episode 的 GT C1→C2 64-update fit 片，固定 probe MSE 仅约 −0.98%/−1.00%；人物/背景可辨，训练前后无明显新增动作或连续性收益。 | 说明此小预算普通 FM 没给出强信号；两片不是同状态动作反事实，也没有 generated-history 测试。不能据此判 H1、H2、模型容量或充分训练的结论。 |

## 当前可作的判断

- **H1（合法动作条件通路削弱）：**代码路径差异与同状态 velocity 干预已有直接证据；行为层面的跨 ABot 机制重复、哪个边组在真实失败状态上主导，仍未知。先测试 H1 的 ABot 复现，避免预设新模块。
- **H2（历史表示与新动作冲突）：**晚切换延迟和 clean/same-σ、自身历史协议差异仅构成假设。尚无在**同一历史来源内部**配对动作、再比较 GT/self 两种来源交互的合格实验；也未排除 H1、自然惯性、不同场景难度和训练不足。当前不应设计“动作感知 Memory”或 SGF+ 作为既定解。
- **数值与协议：**完整 action span、全局 RoPE/native time、Single I0、当前噪声、history 内容、prefix feedback、history commit sigma、SDPA/GEMM 行数会影响归因。先复用[EXP-017 固定布局](../../experiments/EXP-017_v3_fixed_action_layout/judge/FINAL_REVIEW.md)与[EXP-019 真实条件回归](../../experiments/EXP-019_v3_mixed_action_validation/README.md)，不可把不同设置的视频差直接命名为机制。

## 本地可核验工作与文献边界

[H3-World 的本地动作路由实现](../../code/diffsynth_h3_action.patch)可核对已学习的 action–video 信息图；[SolarWM 本地文档](../../../SolarWM/docs/architecture.md)可核对其因果/历史训练设定。AnyFlow、Self Forcing、已有 action-conditioned causal world models 及 action-conditioned memory/cache 更新的**最接近论文和优先权仍需系统外部文献核实**。本地没有检索结果不能推论“此前没有人做”。严格因果分块、KV缓存、LoRA、少步蒸馏或DMD的直接组合都不是本项目已证明的新方法；当前最有论文价值的可能是可重复的动作退化机制与针对它的受控改进，仍待实证。
