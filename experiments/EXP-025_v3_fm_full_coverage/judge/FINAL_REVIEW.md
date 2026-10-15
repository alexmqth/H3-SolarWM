# EXP-025/v1 Judge 最终验收

日期：2026-10-16 HKT。Judge：本会话。决定：**accepted（任务执行与证据）**；模型视觉/动作 **PARTIAL**；新增能力收益 **未证实**；正式V3不升级。

## 审核依据

已读取WORKER_REPORT.md，独立CPU审计账本、冻结来源SHA、固定probe、12原片与6对照完整解码；此前分别核对64/128/256updates checkpoint、参数有限性、optimizer/RNG与训练顺序。零adapter三路逐值相等，完整KV SHA相同；真实C1/C2首步50层路由/梯度与冻结参数门控通过。无Judge GPU调用。

实际570DiT/256backward/256updates/12decode/0encode，11566.613GPU进程秒，阶段与总预算均通过。训练9666.122秒；全程GPU allocated峰24.449GiB，reserved峰28.139GiB。终点RSS66.472GiB不能当峰值。大tensor仅本地保存。

固定16probe总体均值比−0.6011%；C1 +0.2331%，C2 −1.3438%。train/dev C1分别+0.2118%/+0.2553%，C2分别−1.4246%/−1.2542%。与Worker报告一致。均值比与逐项百分比平均不同，不混用。C2较EXP-023的约1.23%下降只有很小差别，不能据此称全层显著更好。

## 完整帧序列结论

四条GT-history审阅全部17新帧及边界，两条self-history审阅全部56帧，详见[连续帧审核](VISUAL_REVIEW.md)和26张图页。人物总体可辨，但self-history运动不足、视点偏移与边界问题未明确解决；前后大部分画面接近。视频收益未证实，与Worker保守结论一致。GT轨迹不是唯一正确未来，没有同状态A/D反事实，不编造方向准确率。

boundary MAD：GT-history 10.49→10.85；self-history 12.41→9.72。后者伴随不同自生成轨迹/运动量，不能作为连续性恢复的充分证据。该判定接受可行性阶段的局部缺陷，不以产品画质作为任务门槛。

## ROI决策与边界

结束此32clip/256updates全层C1+C2候选；不追加rank32、更多epochs、AnyFlow、DMD，不覆盖正式V3。容量、训练块和总更新数同时变化，是联合方案。没有self-history训练；2dev是已观察episode；56帧/8NFE结果不外推到充分训练、长视频或架构上限。

下一任务转向此前已同意但尚未完成的官方SolarWM-H3 Stage2-fix相机基线，先实际补齐最小官方输入并冻结可执行推理方案。语言动作H3-World参考保留；两种控制接口不可直接混成动作对照。

[机器审计](FINAL_AUDIT.json)中的VISUAL_REVIEW_PENDING保留其生成时状态，最终视觉结论以本文与CURRENT_VERDICT.json为准。[汇报与六组视频](../../../report/v3/fm2anyflow/fm_full_coverage/README.md)。
