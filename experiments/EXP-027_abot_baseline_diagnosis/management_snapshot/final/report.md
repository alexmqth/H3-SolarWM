# EXP-027/v2 Worker 报告：ABot 两片 C2 普通 FM 可拟合性

2026-10-16 HKT。**已完成获批的64次 C2 更新、固定 probe 与4条必要对比视频，结果是小幅局部改善，但没有清楚的动作/画质跃升。** 这轮仍处于两片训练样本的可拟合性验证，不能作为同状态动作反事实、泛化或 generated-history 长视频结论。

首次 GPU 尝试在第1次更新前被新 runner 的错误审计拦截：GT C1 历史提交合法使用 `index=0`，新审计却对所有阶段强制 `index=1`。它消耗369.3秒、51次顶层 DiT，0次更新。Judge 审查后签发[补充授权](submission/experiments/EXP-027_abot_baseline_diagnosis/ADDENDUM_01_COMMIT_AUDIT_RECOVERY.md)；独立 `recovery_01/` 入口修正阶段审计，CPU门控通过，并从保存的0-update checkpoint/optimizer/RNG恢复，复用初始40条 probe 和零adapter回归，未重跑前面评测。原失败账本与源码保留。

恢复运行在 GPU0 完成64次更新，两条 fit 片各32次。固定多σ、独立噪声平均 MSE：S+D 从0.183778降至0.181985（−0.98%），A+S 从0.191754降至0.189831（−1.00%）；两条已观察 dev 片分别下降0.91%与0.67%。两条 fit 片均未达到10%的可选 self-history 视频费用门槛。完整逐clip/σ/seed数据见[初始probe](submission/experiments/EXP-027_abot_baseline_diagnosis/PROBE_000.json)、[32-update probe](submission/experiments/EXP-027_abot_baseline_diagnosis/recovery_01/PROBE_032.json)和[64-update probe](submission/experiments/EXP-027_abot_baseline_diagnosis/recovery_01/PROBE_064.json)。

四条必做[GT39+C2视频](submission/experiments/EXP-027_abot_baseline_diagnosis/recovery_01/VIDEO_RESULTS.json)均为56帧、24fps：前39帧是真实历史，后17帧是FM-8NFE模型生成。两条[checkpoint0-vs64并排视频](submission/experiments/EXP-027_abot_baseline_diagnosis/recovery_01/comparisons/)与[接触表](submission/experiments/EXP-027_abot_baseline_diagnosis/recovery_01/CONTACT_SHEET.png)便于查看；[CPU媒体/像素复核](submission/experiments/EXP-027_abot_baseline_diagnosis/recovery_01/CPU_REVIEW.json)保留原始RGB参考和解码验证。生成17帧相对原始视频的PSNR分别提高0.42dB和0.34dB；人物与背景在两版本都基本完整，视觉差异较小。光流幅度未明显降为静止，但不能据此证明动作绑定恢复。两片来自不同初始画面和episode，不能作A/D同状态因果对照。

[合并账本](submission/experiments/EXP-027_abot_baseline_diagnosis/recovery_01/COMBINED_LEDGER.json)：两次尝试累计3,876.4 GPU进程秒、303次顶层DiT、64次backward/update、4次VAE decode，最大峰值分配显存24.45GiB；GPU进程已退出。完整方法、错误、训练顺序、视频SHA与边界见[详细Worker报告](submission/experiments/EXP-027_abot_baseline_diagnosis/WORKER_REPORT.md)，等待Judge验收。本轮没有新AnyFlow或DMD实验，Worker未操作Git、`progress.md`或`next_plan.md`。
