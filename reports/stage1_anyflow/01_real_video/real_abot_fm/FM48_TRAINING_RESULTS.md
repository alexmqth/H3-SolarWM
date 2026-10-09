# 真实ABot普通causal FM：48更新训练完成

完整48次optimizer更新已正常结束，冻结runtime/模型初始化/训练curriculum/checkpoint审计通过。**这只证明训练完成及固定验证loss小幅改善；训练后视频与动作验收仍在执行。**

| 噪声段 | 训练样本数 | 固定验证样本数 | 验证raw FM loss：0→48 | 相对变化 |
|---|---:|---:|---:|---:|
| 低：sigma≤0.240781 | 4 | 0 | 未测 | 不推断 |
| 中：0.240781<sigma≤0.689441 | 35 | 12 | 0.181478 → 0.178895 | -1.42% |
| 高：sigma>0.689441 | 153 | 4 | 0.086299 → 0.081888 | -5.11% |

验证为4个固定held-out片段，均在chunk2，各4个sigma（约0.296187/0.301996/0.592183/0.929706），共16项。日志中sigma、weight、sample type等逐项一致，实际validation noise tensor hash没有保存，因此不夸大为逐bit噪声审计。16项raw loss均下降，但不同sigma之间loss尺度不同，不能用整个训练期间不同样本的loss画伪学习曲线，也不能替代视频质量。

低噪声训练样本仅4/192（2.08%），这是覆盖事实；它不是增加低噪声权重一定改善的证据。当前没有更改sampling shift或loss weight，后续若需要对照必须区分这两个因素。

## 已核查的训练内容

- 真实ABot：16train片段、独立episode的8validation缓存。48更新使16片段×3chunk各覆盖一次；validation本轮固定使用其中4片段。
- Original H3＋发布action LoRA初始化；无旧训练action residual、无AnyFlow、无critic。零输出visual wrapper全程冻结，全block/refiner的rank8 QKVO/FFN bank共208个模块发生更新。
- GT clean history、完整历史梯度、CPU raw KV、RGB一致的dual anchor、causal action rows/feedback；39 RGB帧、12 latent、5+5+2 chunk。
- checkpoint权重hash、optimizer/RNG状态、数据manifest hash及306项冻结runtime检查通过。

## 成本及后续评测

- 整体训练wall time：**10380.96s**（约2.88小时）；48次update计时合计10029.77s。
- GPU peak allocated：**30125.81MiB**（29.42GiB），单GPU0，其他用户进程保留。
- 更新期间记录192次current预测、48次physical clean commit、192次带梯度history重算；不含validation和checkpoint backward重计算，不能把192称作全部模型计算。
- 训练结束后正常启动三路：GPU0两个真实场景GT30/generated30/generated8；GPU4停车场A/D30/8；GPU1固定GT和step00 generated-state geometry。
- GPU1曾不足34000MiB free而等待；随后原队列自行成功启动。拟迁移到GPU2的交接前检查发现已运行，于是取消迁移，没有重启GPU作业。

[完整训练审计](completed_training_audit.json) · [逐项验证CSV](validation_after48.csv) · [GPU评测队列快照](post48_queue.json) · [CPU报告队列](post48_reports.json)

完整A–D目标尚未完成。B需要真实视频及严格动作结果，C/D仍遵守可信局部能力门槛。
