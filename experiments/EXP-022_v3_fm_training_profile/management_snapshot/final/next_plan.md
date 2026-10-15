# 当前可执行任务：EXP-023/v1

2026-10-15 HKT。EXP-022工程训练通路已accepted；原Exp Worker立即执行32条真实视频的普通FM小数据拟合。CPU自检、编码pilot、批量编码、128updates及固定前后评测均已授权，正常检查通过即连续执行，不等额外marker。

[任务书](submission/experiments/EXP-023_v3_fm_small_data/taskbook_v1.md) · [授权](submission/experiments/EXP-023_v3_fm_small_data/AUTHORIZATION.json) · [固定选样](submission/experiments/EXP-023_v3_fm_small_data/SELECTION.json)。

- 28train+2dev新增编码，4train复用，先两clip pilot；不编码另6dev。
- 零增量新adapter初始化、普通FM C2训练，32样本×4epochs=128updates，每步重建GT C1 KV。
- 0/128updates：4固定clip GT-history、2dev self-generated-history；全部FM-8NFE，同条件前后评测，共12条视频。
- 上限13200GPU秒、440顶层DiT、128backward/updates、62VAE encode、12decode、600text；训练1卡，编码/评测最多2空闲卡。完整阶段限制/硬停见任务书。
- 不预设128updates收敛，不把profile容量当足够容量；无新增AnyFlow/DMD。

EXP-022：413.200GPU秒，11F/2B/2U，零adapter150项KV与velocity exact；allocated20.293GiB、采样点RSS最大64.704GiB。新增36RGB物化完成，画质/动作改善尚未测。[最终审核](submission/experiments/EXP-022_v3_fm_training_profile/judge/FINAL_REVIEW.md)。

Judge下发后立即消息通知固定Worker并确认开工，Worker完成后消息交回Judge验收与push。
