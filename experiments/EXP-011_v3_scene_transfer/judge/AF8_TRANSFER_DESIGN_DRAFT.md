# 候选下一任务：冻结 AF8 的短程场景迁移

2026-10-11，Judge 研究草案。**尚未立项，无新 GPU 授权。** 在 EXP-011 两场景最终结果验收后决定是否发布独立 EXP-012；不改变当前唯一任务。

## 能改变的研究决策

EXP-007 的 AF8 在停车场续写有限可行，但未体现整体优于普通 FM8。训练只使用有限停车场教师端点。用已经准备好的两个固定其他场景检查同一 checkpoint 的行为，可区分“有限适配至少保留迁移能力”与“对停车场的适配在其他场景明显退化”。本轮不以普通画质缺陷为理由追加训练。

## 最小有效比较

- 每个场景复用 EXP-011 **FM8** 的首39 RGB和 clean C1 latent、native Single I0 fixture、动作/噪声及完整位置布局。
- FM8 对照直接复用 EXP-011 G2 的 AA/AD56；不重跑。
- AF8 唯一权重为 EXP-007 AF2 step32 的配套 QKV/target-time。保持8NFE/native shift2.22，相邻 sigma 作为 target-time，其他 V3 协议冻结。
- AF8 必须以自身权重从共同 C1 clean latent **重新构建 C1 raw KV**，不能拿 FM8 的 raw KV 给 AF8 使用。clean commit 为 sigma=target_sigma=0。
- 同场景两方法共享 C1 latent/RGB、C2动作和初噪声；各自KV由各自模型计算。这是匹配历史的模型方案对照，不是相同 raw KV 单因素实验。
- 每场景只生成 C2 的 AA/AD，各到56帧。不测 AF 首窗、不加 C3、不延长、不挑图或 checkpoint。

## 建议预算与阶段

CPU准备核查冻结配对权重、两fixture/C1来源、student入口、target-time路径、无未来条件和实际调用账本。复用已经核查的 EXP-007 interval_student 与 EXP-011 场景输入工具，独立runner/config/source清单，不能修改冻结父实验。

如批准，两个场景各1次clean commit和2×8采样：**34 forward =32sampling+2commit、4 decode、0 encoder、0 update/backward，≤0.35 GPUh**。每卡allocated≤44GiB，磁盘≥60GiB，09:00 HKT截止；使用真正空闲GPU。每场景单独放行，不自动重试。预计新增两套C1 KV约12.7GiB加小型端点；仅对外提交日志/指标/视频/摘要。

## 验收与停止

逐场景检查AA/AD全部34新帧、原分辨率边界和代表帧；人物/场景、动作切换、ghosting与边界分开判断。沿用EXP-011中央ROI中值水平流16步累计作为辅助，不与旧停车场flow量纲混用。真实检查模型各自cache、旧39RGB不变、8NFE网格、target-time、完整56帧解码、时间和内存。

持续主体分解/场景全噪声、协议错误、预算/截止/磁盘阈值触发则停止对应运行，保留负结果；另一固定场景是否继续由Judge决定，不为挽救结果追加训练。普通缺陷PARTIAL。若没有稳定联合收益，冻结AF8作为研究候选并明确普通FM8优先；若有清楚收益，也只形成下一次多样化训练设计的依据，不宣称广泛泛化或成熟质量。

## 依据与限制

父证据：EXP-007/judge/AF3_REVIEW.md 与 EXP-011/judge 的最终输入/生成验收。两张初图来自已存在的ABot validation固定episode，不能声称统计意义上完全未见；有限停车场训练来源和新场景关系必须可追溯。普通FM8和target-time student分别记录，训练与模型条件共同变化，不作纯训练收益归因。
