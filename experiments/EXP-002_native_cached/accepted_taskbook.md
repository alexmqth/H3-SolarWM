# EXP-002：Native条件下的严格缓存视频候选

发布：2026-10-10（Asia/Hong_Kong），Judge。EXP-001/v3已验收并推送GitHub：679258c76e0ceaa2ec54912cc62f1df0e5304e05。**本文件是当前唯一GPU任务授权。**

## 1. Task ID / Version / Status

- Task ID / Plan Version：EXP-002 / 1。
- Research Track：Mainline / V3 candidate；不是正式V3版本。
- Parent Version：Original H3 + released action LoRA；继承V2b的Single I0/native条件与12→5分块，复用既有current-prefix候选工程。
- Worker Status：running（AA/AD第二块56帧已完成，续第三块）；Judge Acceptance：pending。

## 2. Research Question / Hypothesis / Baseline

唯一问题：严格chunk因果、private current prefix反馈与真实persistent video KV，在native条件下能否生成具有可用人物结构和持续A/D响应的自身历史续写？

假设：保留原生初图/时间/位置和首窗长度后，既有current-prefix候选可能形成可用的无训练缓存起点。该假设未证实；若视频失败，停止无训练拓扑微调；依据失败是否有可修复信号决定有限适配或直接换路线，不默认追加训练。

基线复用V2b AA与AD的56/73帧已有视频、同first12_A历史与噪声。AA有EXP-001124帧证据；AD仅有73帧，明确其范围，不能把DD当成AD匹配基线。Original持续A片可作AA参考，Original A→D匹配片缺失如实标记，本轮不新生成基线。

历史证据：`submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/`已经实现本候选连接。旧RGB-dual/5latent的18固定状态诊断只恢复首块Original身份，后续动作delta cosine未恢复，且没有候选生成视频。本轮新内容是native Single I0/12→5条件下的实际cached rollout；不得将旧方案当作首次提出或正结果，不重复18点诊断。

## 3. Controlled / Changed Variables / Inputs

保持Original权重与released action LoRA、停车场832×480、seed13、full37初始噪声、Single I0、native时间、全局RoPE、固定audio条件、30步/chunk、shift2.22、既有实际精度/backend与offload。

联合变化：Same-σ全历史双向重算改为严格chunk因果、sigma0/native条件clean-commit的历史hidden KV；current action own routing/feedback；公共非动作prefix只读取当前chunk video；每个chunk拥有独立prefix，不追溯更新祖先KV。不能声称这是单因素消融。

输入：EXP-001冻结manifest、V2b旧first12_A endpoint、匹配full37 noise/conditioning；旧`current_prefix.py`与测试；`H3-World/code/causal/h3_cached.py`。首12复用仅在空cache时候选图/条件与旧局部Original一致后成立；第二块以后必须本候选自己的输出，不得接V2b后续块或teacher重置。

## 4. Execution Plan

### S0：最小实现与正确性

在独立`submission/experiments/EXP-002_native_cached/`实现cached interval入口。全局(start,stop)与cache chunk index分离，支持[0,12)、[12,17)、[17,22)；不能沿用`index*chunk_frames`作为不等长分块的全局位置。Single I0、`fixed_prefix_timesteps=False`，未来video/action在进入模型前物理移除。

复用既有current-prefix实现。公共prefix不读历史video KV；current action只反馈自己控制的current video；current video读取历史video KV与自己的action。历史缓存只读，每个祖先用其生成时独立prefix/native条件提交一次。commit后仅追加新chunk，禁止每步重算历史Transformer冒充cache。

CPU检查覆盖本轮改动的全局位置/action对齐、未来隔离、cache不变、首12空cache身份、每块一次commit与逐祖先重放参考。已有未改动测试优先复用；不开展后端bitwise一致工程。最多4次真实模型回放用于首窗/缓存复用必要核对，记录实际误差和判定依据；明显图或条件不一致必须修复或报告，不能以ROI豁免。

### S1：共同A-history分支AA/AD，各续第二块到56帧

两条都复用同一个first12_A、自身生成首39RGB和相同初始noise。各自按完全相同的sigma0/native条件prefill，核对两条初始历史KV摘要一致；唯一分支变化是当前[12,17)动作A或D（AA/AD），30步采样只读同语义cache。这样直接检验固定历史下的当前动作响应，避免两个不同历史的差异冒充控制效果。冻结已有首39RGB，只追加新17帧。保存cache摘要/字节数、输入hash、采样/搬运/VAE时间及原片，与V2b相同范围对比。

若严重结构崩坏或明显持续控制失败，停止受影响路径并提交有限负结果；不要追加mask、gain、history-sigma消融。普通边界跳变、模糊或辅助flow轻微反号允许记录限制后继续。

### S2：有信息价值的路径再续第三块到73帧

同一候选协议将自己的第二块sigma0 commit一次，继续[17,22)30步。不得重新接V2b历史、清空cache或重跑首窗。每条最多两个新chunk，总4个；其中一条明确失败不要求另一条也停止。

### S3：报告并等待Judge

提交完整候选片、左V2b右candidate的匹配范围并排片，保留真实边界和失败长度；明确V2b自身历史与候选自身历史后续不同，不描述为同状态反事实。静态帧序列与实际播放检查分开记录。

建议入口（由Worker实现后冻结实际命令）：

```bash
.venvs/h3world/bin/python submission/experiments/EXP-002_native_cached/run_rollout.py --config submission/experiments/EXP-002_native_cached/config.yaml --path AA --gpu <idle_gpu>
```

AD同理（共同A首窗，当前及后续D）；runner应先产出第二块供检查，再显式恢复第三块，不自动排队新实验。最终报告提交后停止GPU扩展。

## 5. Resource Budget

- 0 optimizer updates；两条当前动作分支AA/AD（首窗共同A，其后持续A或D）、一个scene/seed。
- 新sampling≤120（4×30）；first12 prefill与second-chunk commit合计≤4；必要模型回放≤4；完整denoiser前向总≤128，失败/重试也计入。
- 新VAE decode≤4，0 RGB re-encode；复用首39已发布RGB。
- 全任务≤3 GPU-hours且首个GPU进程起≤4h elapsed，计入加载、失败、commit与活跃进程；每卡allocated≤44GiB。
- 项目总GPU：2026-10-10 09:00 HKT前≤8，之后≤3；08:30起不新增超过3卡的占用，09:00前释放多余卡，之后不自动恢复8卡。本任务通常2卡即可。

## 6. Evaluation / Acceptance

分别报告：执行有效性、strict causal/persistent KV实现、动作响应、人物/场景结构、效率测量。辅助flow/cosine只作解释，实际视频与主要结构决定能力范围。

只在同一个候选中确认历史未来隔离、cache真实复用、动作与视觉共同可用，才接受为值得延伸的V3候选。首12身份不算新的能力收益；实现正确不代表视频通过；本轮73帧不覆盖124帧或跨场景。

计时包含prefill/commit、sampling、CPU-KV传输、VAE、编码；报告每块cache tokens/bytes及GPU峰值。匹配硬件/offload与计时范围后讨论观察成本，不将理论KV直接写成实测加速。

执行正确且结果为负也可accepted，模型能力按PASS/PARTIAL/FAIL真实记录。有限片段已有充分证据决定下一步时收敛，不要求补完所有内部机制。当前为可行性验证，接受可辨动作与基本结构伴随模糊、节奏不均及普通边界缺陷；不以高保真或动作自然度作为暗加门槛。

## 7. Stop Conditions / Out of Scope

协议/输入错误、未来泄漏、历史回改、NaN/OOM、预算超限、严重结构崩坏或持续控制失败时停止相应运行；向Judge交付事实。不得自行提高预算。

不做训练、额外场景/seed、124帧扩展、额外切换序列（已授权AA/AD共同首窗分支除外）、超参数搜索、AnyFlow/DMD、重复18点诊断或新增基线生成。

## 8. Deliverables / Next Decision

独立实验目录README/config/源码manifest/metrics/MANIFEST、实际命令日志、视频、cache正确性与资源证据；根report.md标明实际状态并通知Judge。Judge审阅、归档、更新submission并同步GitHub后发布下一任务。

- 若同协议动作/结构可用：下一任务做124帧自身历史和效率验证，再判断V3正式验收。
- 若失败：停止此无训练方向；只有存在明确可修复信号时才设计一次有限适配，否则归档并换更有依据的路线。不默认训练挽救，也不继续零散mask消融。

本任务授权限于上述实现与≤128次前向。完成或停止后提交report，由Judge验收；不得自动延长。
