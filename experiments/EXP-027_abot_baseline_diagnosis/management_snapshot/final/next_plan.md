# 当前状态：EXP-027已验收收口，无运行中GPU任务

2026-10-16 HKT。两片ABot C2普通FM完成64updates、0/32/64固定probe及四条必要前后视频。Judge接受执行与证据，新增动作/视频收益未证实，正式V3不升级。

[视频与成本](submission/report/v3/fm2anyflow/fm_two_clip_fit/README.md) · [最终审核](submission/experiments/EXP-027_abot_baseline_diagnosis/judge/FINAL_REVIEW.md)。

## ROI决定

固定MSE两fit片仅下降0.975%/1.003%，逐帧前后接近，不围绕这一收益继续增加同配置updates/rank/种子。停止当前两片GT-C2短预算候选。64updates不足以否定充分训练，也没有self-history/动作切换结论。

## 后续方向与当前执行边界

仍聚焦ABot/H3-World语言动作与V3 causal/KV。后续若要推进基线训练，须先说明相较EXP-023/025/027新增的可验证依据、代表性动作/episode覆盖、训练更新预算与中途学习曲线停止条件；不能机械追加一轮同类小步微调，也不能直接宣称需要AnyFlow/DMD/Memory。本轮没有自动授权新GPU任务或新实验编号。

SolarWM官方参考保持停止，不查权限/下载/编译，不作为ABot前置。原Exp Worker交付后保持收口。累计本轮3876.365GPU秒/303DiT尝试（含1失败）/64updates/4decode，GPU0已释放。全部原始失败与修复证据保留。
