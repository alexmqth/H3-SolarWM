# EXP-027：ABot 两片C2普通FM拟合验证

当前阶段：v2首轮单卡运行被新 runner 的阶段审计错误中止，Judge复核并签发补充01后，独立 `recovery_01/` 完成64次C2普通FM更新、三阶段固定probe和4条必要fit视频。两条fit片平均MSE分别仅改善约0.98%/1.00%，生成后17帧相对原始RGB的PSNR提高0.42/0.34dB；画面结构基本完整，但没有清楚的动作/画质跃升。dev self-history视频未达到预设费用门槛，未生成。Judge已验收执行与证据；新增动作/视频收益未证实，当前候选收口，不升级正式V3。

目的：在H3-World语言动作与V3 causal/KV协议上，区分小预算训练尚无学习信号与自生成历史/动作切换问题，确定值得投入的下一项实验。ABot为项目现有Gameplay数据来源；GT为真实视频历史；FM为Flow Matching，NFE为每chunk推理网络求值次数，training updates单独计数。

[当前Worker报告](WORKER_REPORT.md) · [本轮报告快照](management_snapshot/final/report.md) · [并排视频](recovery_01/comparisons/) · [接触表](recovery_01/CONTACT_SHEET.png) · [媒体/像素复核](recovery_01/CPU_REVIEW.json) · [合并账本](recovery_01/COMBINED_LEDGER.json) · [原失败账本](GPU_LEDGER.json) · [恢复运行账本](recovery_01/GPU_LEDGER.json) · [任务书](taskbook_v2.md) · [补充授权](ADDENDUM_01_COMMIT_AUDIT_RECOVERY.md) · [v1 Judge审核](judge/V1_REVIEW.md)。复用EXP-023/025的实际数据与证据，SolarWM官方参考不是前置条件。

[最终Judge审核](judge/FINAL_REVIEW.md) · [完整视频与成本展示](../../report/v3/fm2anyflow/fm_two_clip_fit/README.md)。
