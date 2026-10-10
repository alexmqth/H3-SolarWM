# 当前项目进展

更新：2026-10-10 HKT，Judge。当前为可行性验证阶段，优先能改变研究决策的有限实验。

## 1. 正式结论

**V3 Original Feasibility Baseline继续作为已验收正式参考。** EXP-002/003使用Original H3 + released Action LoRA、Single I0、native timestep、own-action/action feedback/current非action prefix feedback、30步Global RoPE、strict causal及persistent raw KV，AA/AD自身历史六块124帧，零新增训练。

AA约RGB79–86有明显瞬态人体形变，后续恢复；连续性与画质PARTIAL。单停车场seed13，无成熟质量、跨scene/seed或公平完整E2E加速结论。旧实验末C6未commit，真实淘汰和超过六块能力尚未验收。

## 2. 版本与证据

| 版本 | 已有结果 | 限制/当前身份 |
| --- | --- | --- |
| V0 Original H3 | 原始双向动作/视觉参考 | 非严格因果生成 |
| V1 Native causal | 分块因果/persistent KV工程原型 | 动作与视觉未同时通过 |
| V2a RGB Anchor | 124帧基本结构 | 动作失败，20秒后段崩坏 |
| V2b Same-σ | EXP-001持续AA/DD124帧动作/基本结构可行 | 双向历史重算，无persistent KV，切换PARTIAL |
| V3 Baseline | EXP-002/003 AA/AD124帧严格因果/KV/动作/基本结构可行 | 正式参考，冻结 |
| V3-SW-G | 显式长分块、Global位置、最近5祖先真实淘汰的CPU实现 | EXP-005阶段一accepted；GPU能力NOT_TESTED，优先候选 |
| V3-SW-L | 同窗口，video Sliding Local位置读时重映射 | 独立CPU候选，非历史重算等价，GPU能力NOT_TESTED |

短暂V2c分类按用户最新决定恢复为V3 Baseline。V2a/V2b与V3没有顺次权重继承关系；SW候选不得覆盖Baseline。目录见[V3家族](submission/report/v3/README.md)和[V2家族](submission/report/v2/README.md)。

## 3. 阶段一历史：EXP-005 / v1

**阶段一CPU交付已由Judge接受；以下为v1历史，当前v2 GPU状态见文末。** 14项CPU测试通过，Judge独立复跑3.74秒。真实H3 cache/router/RoPE与小张量验证精确祖先、淘汰、容量、旧区间attention逐元素一致、实际模型传参、未来行裁剪、clean commit、Local位置/prefix保持及RGB append-only。没有加载33B权重，0 GPU forward、0VAE、0训练、0新视频。

[Worker交付](submission/experiments/EXP-005_v3_sliding_window/README.md) · [Judge审核](submission/experiments/EXP-005_v3_sliding_window/judge/FINAL_REVIEW.md) · [三个候选与风险](submission/experiments/EXP-005_v3_sliding_window/JUDGE_PROTOCOL_REVIEW.md)。

**尚缺经认证的>37latent原生输入和生产GPU runner。** 默认长packed构造会移动已有prefix/video位置；现有长输入只用于CPU结构测试，post37真实GPU调用被禁止。审批前须冻结新增action/位置/噪声，保持原37条件逐值不变，并准备逐调用预算账本。CPU通过不构成完整模型或生成能力通过。

## 4. 下一步与资源

[分阶段GPU提案](submission/experiments/EXP-005_v3_sliding_window/GPU_PLAN.md)：输入/runner就绪后另批G0首次淘汰前回归，再SW-G第7/8块，再单独决定SW-L。核心上限281forward、9VAE、1.70GPU小时、1卡；C9默认关闭。本次仅提交提案，当前GPU额度仍0。项目自2026-10-10 09:00 HKT后≤3卡，不每晚自动恢复8卡。

只验证历史video KV有界；已知action prefix、latent/RGB保存和VAE全前缀解码仍可能增长，不称整个系统无限生成。按可行性接受普通缺陷，持续严重失败则停止方向，不做低收益扫参。

后续[V3-FM8/V3-AF研究设计](submission/experiments/EXP-005_v3_sliding_window/FUTURE_ANYFLOW.md)分别检验全程普通FM8步与新的target-time-conditioned finite-map student；匹配NFE/历史、各模型用自身权重构造cache。旧协议AnyFlow checkpoint不代表V3-AF完成。本轮不训练或DMD。

## 5. 已完成任务与成本

| 任务 | 正式结论 | 实际增量资源 |
| --- | --- | --- |
| EXP-001 | V2b持续AA/DD124可行性，切换PARTIAL | 见原任务账本 |
| EXP-002 | 73帧严格缓存与同历史A/D响应 | 与EXP-003合计310forward/10VAE/约0.66597GPU小时，复用首窗成本另计 |
| EXP-003 | V3 Baseline AA/AD124可行性 | 186forward/6VAE/0训练/0.436741GPU小时 |
| EXP-004 | 原权重8步续写AA/AD73可行性，画质PARTIAL | 34forward/4VAE/0训练/0.129449GPU小时 |
| EXP-005阶段一 | CPU实现/协议准备accepted；GPU未批 | 14 tests，0GPU/0VAE/0训练 |

EXP-004首39帧复用30步，尚无全程8步或AnyFlow完成结论。视频见[8步证据](submission/report/v3/v3_baseline/8step_continuation/README.md)。EXP-003最终CPU历史cache为18.131GB，增量sampling成本与V2b比较包含协议变化，不能归因纯KV或称公平E2E速度比。

## 6. 交付记录

正式V3发布`d039352941708d4c7c757d696009b67c3e0e2468`；EXP-004结果`4a90b78623a728502347287d9dc0fff290bbb998`；短暂V2c分类`575530a3b5d660f8ac3559502a848d08c3e2a1b1`均为历史已推送提交。当前恢复记录见[迁移核验](submission/archive/v3_baseline_restore_20261010/README.md)，原始历史见[archive.md](archive.md)。本轮恢复V3分类与EXP-005阶段一提交`bd5711d18b4d35ec8711d52c39c205bd2a96264e`已正常推送origin/main；远端SHA一致，submission工作树干净。457个现行本地链接、13项源码hash与107项冻结证据检查通过。

## 最新执行授权：EXP-005/v2

用户已明确批准GPU实验。Judge批准G0真实模型回归；G1在G0及长输入认证通过后放行，L1独立决策。核心上限281forward/9VAE/1.70GPU小时，单卡、项目≤3；C9关闭。G0已在GPU0启动，47latent真实fixture已通过CPU认证；尚无GPU能力结论。最新执行以[next_plan.md](next_plan.md)为准，前文阶段一状态为冻结历史。

G0真实GPU回归已验收：两组velocity、最终latent、全部124RGB均与冻结参考一致；34forward/1VAE/0.085864GPU小时。G1第7/8块Global两动作路径已放行，生成能力结论待实际结果。

G1已完成并由Judge接受158帧有限可行性：A/D方向相反、切换响应可辨，真实淘汰后的video KV保持14,164,800,000 bytes；人物/场景基本可用，D-C8持续拖影与边界跳变记为quality PARTIAL。实际123forward/4VAE/0.296852GPU小时。L1进入最终manifest准备，待绑定授权执行既定对照；不扩展C9。
