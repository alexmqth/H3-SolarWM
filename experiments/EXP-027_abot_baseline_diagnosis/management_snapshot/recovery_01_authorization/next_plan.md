# 当前执行：EXP-027/v2 ABot两片C2普通FM拟合验证

2026-10-16 HKT，Judge已审核CPU盘点并授权原Worker立即实现/执行。CPU来源与零入口回归、首步梯度检查通过后自动连续运行，不等待额外marker。

[任务书](submission/experiments/EXP-027_abot_baseline_diagnosis/taskbook_v2.md) · [授权](submission/experiments/EXP-027_abot_baseline_diagnosis/AUTHORIZATION_v2.json) · [Judge审核](submission/experiments/EXP-027_abot_baseline_diagnosis/judge/V1_REVIEW.md)。

两条既有S+D/A+S片，从EXP-025零adapter初始化，full50 QKV rank8、C2 GT-history普通FM，64updates/各片32次。0/32/64四片×5σ×2独立noise固定probe；正常完成后必须两fit片前后4视频，强学习信号时可选dev self-history2视频。10%只控制可选视频，不是质量验收门槛。

硬上限单空闲L40/6000GPU秒/336顶层DiT/64updates/6decode/0encode。原提案漏计12probe commit和6回归已补齐。为必要视频预留800秒；无自动重试/追加训练。正式V3冻结。

SolarWM官方参考停止，不查权限/下载/编译，不作为ABot前置。此次目标是局部可拟合性，不是泛化或同状态动作因果对照；64updates无收益不能否定充分训练。

## 补充01：首次训练前审计索引误拒绝，已授权一次最小恢复

首次0updates停止：369.255秒、51DiT尝试，零回归与40初始probe完成。新审计错误要求历史commit也index1；正确为commit0/forward与backward1。冻结旧证据，recovery_01修复实际helper并CPU检查后直接恢复，不重复初始评测。新预算≤5500秒且累计≤6000秒，新增286DiT/累计337含失败1，64updates/最多6decode不变。无后续自动重试。[恢复任务书](submission/experiments/EXP-027_abot_baseline_diagnosis/ADDENDUM_01_COMMIT_AUDIT_RECOVERY.md)。
