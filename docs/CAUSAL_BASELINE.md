# 单独验收 H3-World 因果化：保存的版本与当前边界

2026-10-09。**因果执行和缓存工程已经完成，但尚无一个完整自由 rollout 版本同时通过画面与动作验收。** 本页把 AnyFlow/DMD 之前的证据单独固定下来；保存已有模型配置、adapter、代码和完整视频，不把“能执行”写成“质量已通过”。逐文件哈希及配置见 [causal_baseline.json](../reports/stage1_anyflow/07_protocols/causal_baseline.json)。

## 按研究目的分成四步

| 步骤 | 本项目使用/研究什么 | 当前状态 |
|---|---|---|
| 1. 双向预训练底座 | 已发布 MiniMax-H3 基础权重＋H3-World action LoRA | 直接使用，不重新预训练；这是本项目的选择，不宣称逐项复现 SolarWM Stage0.5 |
| 2. 可信的 causal H3 | 当前窗口接收动作，跨窗口只访问过去；先以普通 FM/足够采样步数检查画面、控制和历史衔接 | 执行/缓存通过，动作＋画面联合门槛未过 |
| 3. AnyFlow few-step | 在已经可信的因果生成函数上学习有限区间映射，降低每块采样次数 | 两条伪标签视频、最多136更新的历史试验未通过；不能视为因果适配的质量已经成立 |
| 4. On-policy DMD | 在 student 自生成轨迹上，以真实/生成分布的 score 差做分布匹配 | 用于研究 generated-history 差距；不保证自动修复局部动作错误或视频质量 |

这里把因果适配与 AnyFlow 拆开，是为了定位问题；SolarWM 官方 Stage1 的配方包含 causal teacher forcing 和 AnyFlow，并不等于官方另有一个完全独立的“causal-only stage”。DMD也涉及少步生成分布的匹配，不能简化成必然有效的后处理修复。

## A：画面和动作都较好的短窗口——作为正控保存

**[播放：39帧完整 A/D 窗口](../reports/stage1_anyflow/02_causal_diagnostics/local_topology/native_single_anchor12_calibration/AD_window12.mp4)**

配置为 Original H3＋released action LoRA、原生 text/action 时间、单张初始图像、12 latent 当前窗口、30步、seed13；没有新增训练、AnyFlow 或 DMD。A/D flow 为 +1.250421 / −0.977313。已有全39帧静态评审认为人物和停车场结构完整，动作明显不同。

**它只证明无历史的当前完整窗口可工作。** 窗口内部动作预先已知，没有跨窗口生成历史，也没有 persistent KV；不能把这个好视频命名为“causal H3 已通过”。[条件校准结果](../reports/stage1_anyflow/02_causal_diagnostics/local_topology/CONDITIONING_RESULTS.md)记录了它与5-latent分块失败的区别。

## B：最接近局部目标的跨窗口候选——作为部分成功保存

候选 ID：`original_local_N_30step_reference_history`。这是 **Original 权重＋局部 T2 重算＋同sigma历史条件 N**，零 optimizer；不存在需要另导出的新 trained adapter。对应版本必须连同推理实现、时间/位置条件、历史处理及输入状态保存，只有权重文件不能代表该协议。

配置：当前12 latent，保留局部原始 directed action/video 信息流，未来窗口 action/video 在 refiner 前删除；30steps/current window、shift2.22、seed13、native text/action time、原始单I0。保存的过去状态只读，临时将其按当前sigma加噪；每sigma重算可见hidden，**没有 persistent hidden KV 复用**。

| 固定参考历史 | 当前 A flow | 当前 D flow | 局部画面 |
|---|---:|---:|---|
| Original A-history | +1.645495 | −1.546733 | D较完整，A末段腿部残影 |
| Original D-history | +0.614104 | −0.673822 | D较完整，A在RGB59–68有明显手臂/躯干重影 |

**[固定A-history完整对比](../reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/window1_A/CN_AD_context.mp4)** · **[固定D-history完整对比](../reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/window1_D/CN_AD_context.mp4)**

两条视频都为：左clean历史C，右同sigma历史N；上当前A，下当前D；前8帧共同历史＋42帧当前生成。保留两个history及A/D全部分支，不只展示较好的D。历史由Original生成并固定，**不是该候选连续生成自己的history，不是真实GT，也不是50帧自由rollout**。

可单独查看局部完整结果：[A-history→A](../reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/window1_A/A.mp4)、[A-history→D](../reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/window1_A/D.mp4)、[D-history→A](../reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/window1_D/A.mp4)、[D-history→D](../reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/window1_D/D.mp4)。每条42当前帧，约1.75秒。

结论是局部方向响应有所恢复，人物结构未过；两个history、一个后续窗口不能证明长期稳定性。Farneback水平flow是运动代理，不是角色动作准确率。原 [VIDEO_RESULTS](../reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/VIDEO_RESULTS.md)、[协议](../reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/protocol.json)及全部失败画面保持原样。

## C：完整124帧缓存生成——两个已有adapter版本均保留

**[播放：Original / 动作较强旧版 / 视觉较完整新版，A/D三列对比](../meeting/action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4)**

| 版本 | 保存的adapter | 结果 |
|---|---|---|
| 旧 fixed-mix | [action_adapter.pt](../checkpoints/legacy_fixed_mix/action_adapter.pt)；[对应配置](../checkpoints/legacy_fixed_mix/README.md) | A=+0.1427、D=−0.3105，方向符号正确但响应较弱，后段撕裂/重影 |
| RGB visual | [causal_adapter.pt](../checkpoints/visual_rgb_tail16/causal_adapter.pt)＋[action_adapter.pt](../checkpoints/visual_rgb_tail16/action_adapter.pt)；[对应配置](../meeting/DEMO_PROVENANCE.json) | 124帧结构相对更完整，A=−0.7841、D=−1.0075，A符号错误 |

这两版都使用自己的 generated history、CPU raw KV、5-latent chunks，8steps/chunk×8chunks，64 noisy forwards＋8 clean commits。它们没有使用AnyFlow训练；“8步推理”本身不是“已完成few-step distillation”。RGB版本是普通FM/replay＋endpoint辅助，不是DMD student update。两版配置有多项差异，不当作单变量对照。

**它们已经保存，但都不能作为“画面与动作都合格”的训练起点。** 这些124帧指标与B的局部42帧指标协议不同，不能按数值大小直接排名。完整复现还依赖外部H3权重、released action LoRA和DiffSynth实现，见 [REPRODUCE](../REPRODUCE.md)。B的历史源tensor与冻结runtime仍位于工作区outputs；Git归档保留脚本、协议和哈希，不是脱离外部输入即可运行的全量模型包。

## 下一步应通过什么，才进入AnyFlow

当前保持研究冻结。本次只整理和固定已有证据，没有新增GPU实验。

若恢复研究，优先解决步骤2：在同一可信 GT/reference history、同一当前noise和充分采样步数下，多个chunk上的A/D反事实既有正确方向，也不出现严重人物分解，并保持历史衔接。通过后再比较因果30步与AnyFlow4/8步，最后测generated-history差距和on-policy DMD。进入AnyFlow不要求先解决20秒自由生成的全部漂移，但目前局部结构这一项还欠缺。

**当前保存状态：有明确的短窗口正控，有值得继续研究的局部候选，有可复现的完整缓存rollout；尚无通过联合验收的 causal H3 checkpoint。**
