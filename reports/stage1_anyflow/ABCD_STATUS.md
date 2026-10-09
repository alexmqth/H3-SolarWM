# 旧A–D工作流：证据归档（已被三实验计划替代）

2026-10-09 02:57：用户明确要求停止继续扩展本工作流，改为[E1局部动作信息流→E2动作后果监督→E3 AnyFlow/Stage2](three_experiments/PROTOCOL.md)。本页保留证据，**不是已完成完整A–D，也不是后续自动执行清单**。旧density全部评测已结束且FAIL，无运行中的原队列；[终态](fm_density_control/FINAL_RESULTS.md)。下方运行状态只适用于其日期。

2026-10-09 02:32：FM密度对照六条自然39帧视频已全部评审，generated30/8仍出现人物分解或背景重影，未通过。固定generated历史12点delta cos仍仅0.035019。停车场A30 flow−0.190239，D及8步原队列继续；完整A–D未完成。 [自然视频结论](fm_density_control/NATURAL_VIDEO_RESULTS.md)。C新增[512样本权重/输出梯度核查](adaptive_weight_audit/README.md)，不是实际参数梯度/训练效果。

最新2026-10-09 00:03：B[单变量密度对照](fm_density_control/README.md)已在GPU1运行4/48，只改采样shift、权重函数保持12，复用旧完成control；后评测队列等原PID结束。旧B验收FAIL仍成立，新效果未知。D新增[DMD完整FMBS梯度接通](dmd_gradient/README.md)，7项CPU检查通过；尚无新的33B on-policy效果，C/D仍未完成。下面是既有证据，不能用较早“无运行任务”描述当前状态。

最新23:35：[同状态动作机制总览](ACTION_MECHANISM_SUMMARY.md)已将用户要求的时间对齐、直接路由、KV与A/D实际效果逐项列明。新增[噪声扫描](fm_noise_audit/INTERPRETATION.md)各54状态完整：整体raw MSE仅改善1.47%，sigma1均值变差，不能归结为单一低噪声缺口。B有限实验验收仍FAIL，C/D前置未通过，未新增训练；两扫描进程已退出。

D新增[共享H3角色实现](shared_h3_roles/README.md)，6项tiny-H3 FP32/BF16测试含非零旧visual/AnyFlow、Original teacher identity、critic更新插入活跃FMBS图后的完整student梯度保持。它推进工程准备，不是33B DMD或视频效果完成。以下保留历史证据；当前状态以本段和表格为准。

最新22:50：B的48更新及全部10条视频/36个同状态geometry已经完成，action/视觉验收FAIL；隔离prefix候选也未恢复后续chunk几何。C/D前置仍未过。见[完整验收](real_abot_fm/FM48_COMPLETE_REVIEW.md)和[机制诊断](current_prefix_candidate/INTERPRETATION.md)。

本页对应用户指定的完整 A–D 目标。只完成诊断或数据准备不能宣称完成这个目标。

| 阶段 | 当前证据 | 状态与下一门槛 |
|---|---|---|
| A：原始 / 未训练 causal / FM / AnyFlow 同状态 field 对照 | 12 个状态、统一 SDPA 后端、相同完整输入范围；未训练 causal 整体余弦 0.996253，动作差分余弦 0.060153；匹配 FM32 为 0.054469、AnyFlow32 为 0.058434 | 完成输入范围受控的诊断；完整历史重算与部署缓存语义的差别单独披露 |
| B：真实视频 causal FM 桥接 | 6个ABot episode；16train/8validation，GT history、RGB dual、CPU KV；48updates和完整10视频/36状态均完成 | 有限预算实验完成，验收FAIL；GT/generated delta cos0.017641/0.009336，parking30/8 A−D−0.018685/0.017456，第二自然场景人物分解。不能由此证明FM不可能；C/D前置仍缺 |
| C：可靠 causal checkpoint 上加入 AnyFlow | A 中未训练 target-time 条件在 r=t 精确保持初始化输出；旧 AnyFlow 实验已保留 | 尚未具备可信的 causal 视频基线。仍需验证训练后对角保持、时间嵌入范围、自适应权重、scheduler 覆盖、flow-map composition |
| D：真正 on-policy Stage2 | 历史critic/DMD smoke未过关；新shared roles＋完整FMBS已通过tiny-H3 CPU图安全验证 | 尚无新33B teacher/critic/DMD/FMBS训练。B/C局部能力可信后再接入分布监督，不能以接口通过冒充效果 |

## 对实验 B 的预注册约束

- Original H3 与原始发布 action LoRA 初始化；不继承此前失败的 visual/action residual。零输出 rank8 QKVO/FFN/refiner bank 为可训练项；33B base 冻结。
- 39 RGB 帧、12 latent 帧、5+5+2 chunks、history=5、causal action rows、action feedback、持久化 CPU raw KV；历史在 sigma=0 clean commit，每个历史 chunk 使用自己的 RGB anchor。
- 普通 FM 的 noise-minus-clean 目标，真实 GT history；完整历史梯度、checkpoint/offload。没有 AnyFlow target-time 模块、teacher 伪标签或 critic。
- 固定 48 optimizer updates、logical batch4、LR 3e-5、seed13；16 个真实片段的每个 chunk 恰好训练一次。保存 0/1/3/16/32/48，不按 loss 下降无限延长预算。
- 沿用 training shift12、validation/inference shift2.22。本轮不同时扫低噪声权重或架构；数据与初始化均不同于旧 FM32，所以不能将前后效果单独归因于数据来源。
- 验证分为真实 GT-history 局部生成和 free-running generated-history；分别检查人物、场景、18–38 帧重影、MAD/boundary、同状态当前 A/D 的 velocity 差分。FM 主验收先用30步，8步为额外诊断；普通 FM 不以直接达到4步为前提。
- 真实片段含联合按键和真实 camera action，不能把 A/D-dominant 样本当成只按 A/D；自然片段之间的光流差不是受控 action differentiation。严格 A/D 控制需要在同图/同状态下替换动作。
- 只有确认 clean-history 局部能力可信，才进入 C/D；不要求 Stage1 提前消灭全部长时 generated-history gap。

## 数据与资源

只下载 6 个公开 episode（视频约480 MiB），不下载500小时完整集合。ABot 来源、修订版本、片段偏移和视频/动作哈希均在 [真实数据记录](real_abot_fm/README.md)。验证与本轮训练 episode 隔离，但不保证与已发布 H3 的原始预训练集合无重叠。

项目最多同时使用3张GPU。旧FM48评测已退出；新增密度分支训练和GPU0几何完成，原GPU1 rollout队列仍运行（以最新快照为准）。其他用户任务保留，单次共享主机时延不能宣称加速。

相关证据：[A完整结果](field_factorization/RESULTS.md) · [之前persistent-KV机制诊断](generated_action_geometry128/INTERPRETATION.md)。

补充证据：[B零更新完整评审](real_abot_fm/BASELINE_COMPLETE_REVIEW.md)。新图Original纯A/D也是弱正控，不能只凭其student输出判定动作保真；保留该结果并补已知正控停车场。

固定generated-state18点也已完成，delta cos0.012540；GT18点为0.017516，首块亦弱一致。见[36点机制报告](real_abot_fm/GEOMETRY_BASELINE_RESULTS.md)。这些是当时的零更新基线；FM48现已完成但未通过，C/D仍未完成。

C/D方法准备：已核查AnyFlow原论文FMBS与SolarWM SGF区别，并完成隔离三段映射的45组官方输出/梯度CPU对照。见[方法与梯度核查](cd_method_audit/README.md)。这不是实际H3的AnyFlow/Stage2训练，C/D仍未完成。

A补充：[首块2×2真实H3路由消融](first_chunk_routes/INTERPRETATION.md)共56前向。现行delta cos−0.0199，单独恢复通用prefix读取video为0.2405，仍未恢复；不涉及历史KV/训练，不改变B协议，也不计为C/D完成。

D生成端准备已前进：[H3 FMBS接入](h3_fmbs_integration/README.md)通过6项CPU小模型测试；这不是33B teacher/critic/DMD训练，C/D仍未完成。B训练总结见[FM48报告](real_abot_fm/FM48_TRAINING_RESULTS.md)。

B首条训练后GT-history已完成，[全39帧评审](real_abot_fm/review/first_trained_gt30/README.md)未见明确视觉质变。该单场景oracle诊断不是完整B、generated-history或纯A/D验收。
