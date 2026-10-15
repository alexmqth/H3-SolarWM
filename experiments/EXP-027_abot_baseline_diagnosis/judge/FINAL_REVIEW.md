# EXP-027/v2 + recovery_01 最终 Judge 验收

2026-10-16 HKT。**Accepted：实验执行与证据完整；有限续写能力PARTIAL，新增动作/视频收益未证实。** 本轮收口，正式V3不升级，不自动增加更新、rank、LR或AnyFlow/DMD。

## 已核验

CPU输入/源代码SHA、首步50层commit/forward/backward及冻结梯度、checkpoint32/64 SHA/形状/finite/optimizer/RNG和训练顺序均独立核对。64updates两片各32次；0/32/64独立噪声同40键probe齐全。未复用失败尝试已消费的训练RNG，恢复从0update保存状态开始，初始probe/零回归未重复。

第一次运行在错误的commit索引审计中停止，0updates；修复只改变新审计phase/index判定，CPU合法/非法helper测试通过。失败51DiT含1次进入后中断，恢复252次，总303次顶层DiT尝试（302完成+1失败），64backward/64updates/4decode，0encode/text。原始失败/补充授权均保留。

总GPU进程费用3876.365秒=64.606分钟=1.07677单卡GPU小时；恢复3507.110秒，其中训练阶段2463.095秒，固定probe591.591秒，视频445.399秒。模型加载/初次回归等费用均计入两次账本。峰allocated跨尝试24.449GiB、reserved29.137GiB；CPU KV6.332GiB。68.147GiB仅为账本终点RSS最大值，非已证实RSS峰值。GPU0已独立确认0MiB，进程退出。

## 结果

固定probe均值之比：fit S+D −0.9752%，fit A+S −1.0027%；已观察dev constant −0.9047%、dev switch −0.6715%。训练低σ误差下降0.839%/0.885%，高σ下降1.782%/1.689%。不能与此前不同probe/样本设置直接作收益排名。

两fit片前后4视频完整交付，全部17新生成帧已观察。[视觉审核](VISUAL_REVIEW.md)显示前后总体接近；人物可辨并有运动，但未确认动作/连续性改善。PSNR20.80→21.22dB和22.71→23.05dB为小幅像素指标变化；boundary MAD无一致收益。可选dev self-history未达到资源开关条件，正确跳过；10%不是任务执行/能力验收硬阈值。

## ROI决策

集中曝光未产生值得直接扩展的清楚收益，停止此两片GT-C2短预算候选。不要重复同类微调追逐约1%的指标变化。结果仍不能区分训练时长、目标、容量及优化的全部影响，也不能否定充分训练后的H3/FM。任何下一轮ABot训练应有新的可验证依据和独立预算；本轮不启动新GPU任务。SolarWM官方参考保持停止。

[机器审计](FINAL_AUDIT.json)生成时标记VISUAL_REVIEW_PENDING，最终结论以本文件与CURRENT_VERDICT.json为准。[视频与成本入口](../../../report/v3/fm2anyflow/fm_two_clip_fit/README.md)。
