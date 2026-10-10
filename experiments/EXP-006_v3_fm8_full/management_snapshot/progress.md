# 当前项目进展

更新：2026-10-11 01:55 HKT，Judge。可行性与ROI优先，普通画质缺陷如实记录。

## 正式结论

**V3 Original Feasibility Baseline继续作为已验收正式参考。** EXP-002/003：Original H3 + released Action LoRA，Single I0/native timestep/own-action/action feedback/current-prefix feedback，strict causal + persistent raw KV，30步Global RoPE，AA/AD六块124帧，零新训练。瞬态人体形变、连续性PARTIAL，单停车场seed13；冻结证据未覆盖。

**EXP-005/v2已完成并审核：SW-G通过158帧有限可行性；SW-L工程通过、生成PARTIAL，无已证实收益，当前无训练Local方向归档。**

| 版本 | 已有结果 | 当前定位/限制 |
| --- | --- | --- |
| V0 Original H3 | 原始双向动作/视觉参考 | 非strict causal |
| V1 Native causal | 分块/KV工程可行 | 动作与视觉未同时通过 |
| V2a RGB Anchor | 124帧基本结构 | 动作失败，20秒后段崩坏 |
| V2b Same-σ | 持续AA/DD124帧动作/基本结构可行 | 历史双向重算，无persistent KV，切换PARTIAL |
| V3 Baseline | AA/AD124帧strict causal/KV/动作/结构可行 | 正式参考，冻结 |
| V3-SW-G | A继续/D切换各158帧，真实最近5祖先淘汰 | 优先滑窗候选；边界跳变、D-C8持续拖影，quality PARTIAL |
| V3-SW-L | 同窗口158帧，动作方向保留、cache正确 | 场景几何/亮度跳变更明显，无联合收益，归档 |

[V3家族](submission/report/v3/README.md) · [V2家族](submission/report/v2/README.md) · [浏览器视频入口](submission/report/index.html)

## EXP-005本轮证据

- 阶段一14项CPU测试已独立复核。显式长分块/精确indices/未来行隔离/Local位置与prefix约束通过。
- G0真实模型两状态旧/新入口差异0；30-step C6 endpoint与全部124RGB逐值复现冻结V3。
- G1首次clean commit C6淘汰C1；C7祖先C2–C6，C8为C3–C7，全部50层检查。历史video KV恒定14,164,800,000 bytes（13.19GiB），旧RGB均不变。
- Global A C7/C8 flow +0.819/+0.768；D −1.575/−1.185。方向和首新增块切换响应可辨。Local相应+1.120/+0.343、−0.598/−1.269；主体仍可辨，但重复几何、亮度闪变更多。
- 同G历史/C8 noisy state的Global/Local velocity relative RMS差异18.25%；历史不变。Local不是无损重定位，也不等价重算历史。

[Judge最终审核](submission/experiments/EXP-005_v3_sliding_window/judge/STAGE2_FINAL_REVIEW.md) · [实际Worker报告](report.md) · [三个候选协议](submission/experiments/EXP-005_v3_sliding_window/judge/STAGE2_PROTOCOL.md) · [完整证据目录](submission/experiments/EXP-005_v3_sliding_window/README.md)

## 成本与边界

本轮单GPU0顺序运行，实际**281forward、9VAE、0训练、0.662707GPU小时**，低于1.70GPU小时；GPU allocated峰值26.621GiB，核心elapsed48.49分钟。EXP-005作业均已结束，无C9/重试。当前用户夜间新授权见下，至10月11日09:00不设人为项目卡数上限，之后最多3卡。

仅验证158帧/24fps约6.58秒与两次真实淘汰；不能声称无限长、跨scene/seed稳定。只历史video KV有界，prefix、latent/RGB保存、VAE全前缀解码仍可增长。长fixture显式保持旧37条件，并非默认H3长packed重建。阶段增量时间不是从零生成全片E2E，也不支持公平speedup排名。

## 当前执行：EXP-007 V3-AF

EXP-006已经完成：**新8步首39+AA/AD73有限可行性通过，quality PARTIAL**。方向C2为+0.846/−0.321、C3为+0.505/−0.826；人物/场景可用，边界跳变与残影保留。43forward/5VAE/0训练/0.136905GPU小时，allocated peak26.061GiB。[Judge验收](submission/experiments/EXP-006_v3_fm8_full/judge/FINAL_REVIEW.md) · [V3-FM8视频](submission/report/v3/v3_fm8/README.md)。不继续扫步数或强行扩长。

当前[next_plan.md](next_plan.md)为EXP-007/v2：真实模型新target-time + last8 rank8 QKV student的初始化回归和单次finite-map update，22forward/4backward/1update/0VAE/≤0.75GPU小时。AF0真实tiny-H3 CPU已通过：普通入口/初始化对角一致、prefix时间正确、有效梯度、反向cache不可变、更新后KV必须刷新。尚无AnyFlow 33B训练/生成验收结论，旧AnyFlow产物不作新V3-AF证据。

用户授权Judge持续到2026-10-11 09:00，阶段完成后自主审核/布置下一项任务，不重复请求用户批准。期间按已批预算使用实际空闲卡，不人为限制项目总卡数，不抢占他人进程；09:00后最多3卡。V3-AF真实warmup通过后再批有限训练与匹配8NFE视频；DMD在基础student可用后独立推进，保留[协议预审](submission/experiments/EXP-007_v3_anyflow/FUTURE_DMD.md)。

## 历史与交付

EXP-001 V2b124、EXP-002/003 V3 Baseline124、EXP-004普通FM8续写73（首窗30步）、EXP-005阶段一CPU均已归档。历史提交与结果见[archive.md](archive.md)。EXP-005提交`7639bea7cc0fb9e3b27295ce87422cf2d06c076e`已正常推送origin/main，本地与远端SHA一致，当时submission工作树干净。505个本地链接、79项源码/基线摘要、45项产物摘要通过；未提交权重、大KV/latent/npy或字节码。

EXP-006正式结果与EXP-007 AF0准备正在进行本轮Git归档，发布后记录准确SHA。
