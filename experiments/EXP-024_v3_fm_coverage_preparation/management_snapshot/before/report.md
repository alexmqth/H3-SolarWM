# EXP-023/v1 Worker 交付：V3 真实视频普通 FM 小数据拟合

2026-10-15 HKT。任务执行完成，等待 Judge 验收。32 train + 2 dev 的 34 条 56 帧真实源 RGB clip 均重新编码；单卡训练 tail8 QKV rank8 adapter 共 128 次更新，随后完成冻结的 checkpoint0/128 匹配评测。没有启动 AnyFlow、DMD 或新增训练分支。

结果：同状态、同噪声的 8 个 FM probe MSE 全部微降，均值 **0.154220→0.152322（-1.23%）**。四条 GT-history 局部续写与两条完整 self-history 的前后视频可播放，但整体视觉和运动接近；两条 self-history 仍明显偏离源视频运动路径，动作切换改善未获证实。训练链路通过与模型能力改善应分别验收。

[完整 Worker 报告](submission/experiments/EXP-023_v3_fm_small_data/WORKER_REPORT.md) · [六组三列视频](submission/experiments/EXP-023_v3_fm_small_data/paired_gallery/) · [数值对照](submission/experiments/EXP-023_v3_fm_small_data/POST_COMPARISON.json) · [完整预算](submission/experiments/EXP-023_v3_fm_small_data/CUMULATIVE_BUDGET.json) · [checkpoint128 身份](submission/experiments/EXP-023_v3_fm_small_data/checkpoint_128.json)。

累计 444/449 顶层 DiT、128/128 backward/update、70/70 encode、14/14 decode、6222.051/13800 GPU 秒；首次真实 checkpoint backward 路由门控通过。三次失败尝试的成本与错误均在完整报告和原账本保留。Worker 未自行提交或推送 Git。
