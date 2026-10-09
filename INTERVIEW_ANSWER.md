# 面试题回答：在 H3-World 中验证 SolarWM 的因果少步生成

**结论：因果分块与持久KV的工程迁移可行，但尚未证明在保留动作控制和画质的前提下提高长视频端到端效率。** 已在真实H3权重上完成124帧及更长rollout；会议主片在124帧上比旧版更完整，但A/D方向验收失败，同checkpoint的20秒视频明显崩坏。后来实现并训练了TF-AnyFlow，也准备了更严格DMD所需的生成梯度与角色隔离；这些不等于完整SolarWM Stage1/Stage2效果复现。

截至2026-10-09，最后一轮E2的FM-only与FM+action各4更新、A-history/D-history和两条真实GT-history评测全部完成。新增动作损失没有一致收益，当前A分支重影仍在，按负结果冻结。不扩到16更新，不重启AnyFlow或Stage2。

## 1. SolarWM各阶段做什么？

这里使用**仓库命名**，与论文的Stage1/2/3编号依次对应。

| 仓库阶段 | 注意力和历史 | 训练作用 | 本项目对应状态 |
|---|---|---|---|
| Stage0.5，Bid-Cam | 双向视频上下文 | 普通flow matching；把预训练模型适配到世界数据与相机条件，作为后续初始化及冻结teacher | 沿用发布的MiniMax-H3与H3-World action LoRA，没有重训SolarWM相机适配 |
| Stage1，TF-AnyFlow | chunk内可见，跨chunk只看过去；teacher-forced clean history | 学任意噪声起止时间之间的flow map，得到少步自回归初始化 | 早期主片只是causal FM/replay；后来真实实现、训练和评测AnyFlow，但4/8步动作与画质门槛未过 |
| Stage2，SGF/DMD | student在自身generated history上rollout | frozen双向teacher给真实分布方向，trainable fake-score追踪student分布，DMD匹配分布；SGF/rollout-and-replay传递含历史KV写入的生成梯度 | 旧Stage2-lite闭环跑通但有teacher topology/endpoint梯度近似；较严格DMD/FMBS/角色隔离只有小H3单元验证，未完成新的33B完整训练 |

SolarWM H3的原生控制是几何相机条件，H3-World额外有逐latent键盘动作和directed attention；所以前者的成功不能直接保证后者的W/S/A/D保真。

## 2. 为什么可以减少采样步数？为什么少步不等于整体加速？

普通FM拟合瞬时速度场，多步solver沿速度场积分。AnyFlow显式学习从`t`到`r`的有限区间映射，使少量较大更新有训练依据。Stage2再让少步student适应自己的生成状态分布，缓解teacher-forced history与generated history之间的差异。DMD的监督来自teacher与fake-score之差，不只是逐点复制teacher velocity。

**KV cache只复用历史计算，不学习少步映射。** 会议主片的8steps/chunk是人为指定的采样预算，该checkpoint没有AnyFlow训练，不能称为成功的少步蒸馏。

- Original：124帧，30次完整序列denoiser forward。
- 主片causal：8chunks × 8steps = **64次局部noisy forward，另8次clean commit**。
- 局部forward更短，但还要支付CPU KV传输、RGB anchor decode/re-encode及VAE等成本，不能用30/8计算speedup。
- 历史主片单次记录：Original约442–454秒，causal约673–768秒。**没有端到端加速证据**，也不是warmup后重复均值。

## 3. 原始H3与因果原型有什么区别？

Original H3在完整目标序列上迭代去噪，action rows通过特定有向边绑定视频时间位置。强行收紧attention会改变action→video以及video→action的多层依赖；仅保留action参数接口不代表保留动作能力。

本项目有两条不同执行路线，不能混为同一模型：

| 路线 | 时间和缓存规则 | 已验证什么 | 代价或限制 |
|---|---|---|---|
| 旧persistent-KV causal原型；会议主片 | 5latent/chunk；当前chunk读过去raw KV；clean commit；滑窗淘汰；CPU offload | 真实权重rollout、缓存生命周期、受控replay一致性 | 动作信息流迁移困难；generated-history漂移；主片总耗时更长 |
| 最新T2/N局部窗口；E1/E2 | 保留可见局部窗口的Original有向关系；逐sigma重算可见history/prefix hidden；不见未来视频或动作；历史临时加噪到当前sigma | 多个后续窗的同状态A/D符号正确；进一步隔离局部结构问题 | **不复用persistent hidden KV**；当前A仍重影；只是局部条件诊断，不是124帧自由rollout |

T2/N的CPU hidden KV为0不表示没有raw history、CPU权重或其它内存。它的重算成本必须纳入未来性能比较。

## 4. 最小causal训练原型与实现位置

复用33B底座与released action LoRA，只训练小adapter。早期以clean历史teacher forcing做普通FM；后续严格控制同状态动作干预，区分action路由、history条件与训练目标。算力有限时用LoRA、CPU offload、detach/replay和小规模数据，不训练33B全参数。

| 功能 | 代码入口 |
|---|---|
| H3 action rows / 原始directed mask | [abot_action.py](code/abot/abot_action.py)、[H3 patch](code/diffsynth_h3_action.patch) |
| causal attention、raw KV、clean commit、淘汰 | [h3_cached.py](code/causal/h3_cached.py)、[causal patch](code/diffsynth_causal.patch) |
| 推理、输入/adapter哈希、耗时/显存 | [benchmark.py](code/causal/benchmark.py) |
| 小H3真实类FM训练smoke | [train_smoke.py](code/causal/train_smoke.py)、[h3_training.py](code/causal/h3_training.py) |
| AnyFlow目标、时间条件、采样与训练 | [anyflow.py](code/causal/anyflow.py)、[train_stage1_anyflow.py](code/causal/train_stage1_anyflow.py) |
| 旧DMD-lite近似 | [stage2_lite_dmd.py](code/causal/stage2_lite_dmd.py) |
| 更严格DMD工程准备，非33B结果 | [dmd.py](code/causal/dmd.py)、[fmbs.py](code/causal/fmbs.py)、[shared_h3_roles.py](code/causal/shared_h3_roles.py) |
| 最新局部拓扑、真实transition监督 | [local_topology.py](code/causal/local_topology.py)、[local_transition.py](code/causal/local_transition.py) |

KV测试核查cached/recompute等价、滑窗淘汰、非整chunk尾部、重复提交拒绝及动态anchor重放。**replay误差为零只验证同一执行定义下的计算一致性，不证明等价于Original双向模型，也不证明视频质量。**

## 5. 哪些实验失败，说明了什么？

1. **动作与视觉取舍。** 旧fixed-mix有更强A/D符号响应但后段漂移；RGB visual主片更完整但A方向错误。adapter、anchor、routing同时不同，不能把改善只归因于某一个模块。
2. **长时自由生成。** 同一RGB checkpoint的20秒片明显雾化/崩坏。generated-history distribution shift是合理待研究因素，但局部GT/reference history也有结构问题，不能把所有失败归因于缺少Stage2。
3. **AnyFlow。** 实际做过有限预算训练及finite/diagonal与teacher endpoint诊断，内部一致性改善没有自动转化为正确动作与画质。完整可用Stage1仍未验收。
4. **动作velocity诊断。** 同一history/noisy state只换当前A/D，整体velocity拟合较高而动作差分cosine很低；说明主损失可以掩盖动作差分问题。双向teacher可能用到student不可见条件，不能无条件强迫逐点复制全部差分。
5. **最终E2。** 同初始化、同4更新，普通FM对比FM＋动作后果排序。A/D方向在局部N已有恢复，但动作项未一致改善人物结构或动作分离度。8个训练microbatch没有纯A/D，部分混有相机控制，监督覆盖和时序仍需审查。小样本负结果不证明所有动作监督无效。

## 6. 展示与最终技术判断

先看[Original vs causal主片](meeting/annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4)，再看[同checkpoint的20秒完整失败片](meeting/long_horizon/original_vs_rgb_visual_W_20s_481f.mp4)。主片使用**2026-10-06 `visual_online_rgb_tail16_endpoint_ad2`**，包内为`checkpoints/visual_rgb_tail16/`；不是最新AnyFlow或E2结果。[checkpoint与视频对应表](meeting/DEMO_PROVENANCE.md)提供哈希和配置。

值得继续研究的是：先在不泄漏未来的局部窗口中保留H3动作信息流，建立可靠的同状态动作后果监督；只有局部30step动作和结构成立，再评AnyFlow少步与Stage2 on-policy适应。persistent KV的扩展性有价值，但action-dependent prefix可能需要重算，不能为了缓存而假定依赖关系不变。

> We demonstrated executable causal chunk rollout and persistent KV caching on H3-World. We have not established simultaneous action preservation, stable long-horizon video quality, or end-to-end speedup. Local history-conditioning improves action signs, but visual artifacts remain and the controlled four-update action-loss trial adds no consistent benefit. Reliable local causal generation should precede renewed AnyFlow and on-policy Stage2 experiments.

[实验报告](docs/EXPERIMENT_REPORT.md) · [复现步骤](REPRODUCE.md) · [5分钟答辩](meeting/MEETING_SCRIPT.md) · [E2完整证据](reports/stage1_anyflow/01_real_video/real_transition_windows/FINAL_RESULTS.md)
