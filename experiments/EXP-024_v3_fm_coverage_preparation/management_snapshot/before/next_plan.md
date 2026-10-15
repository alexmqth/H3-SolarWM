# 当前可执行任务：EXP-023/v1

**EXP-023补充02/EXP-022验收更正（优先于早先工程PASS）：** 已确认022的current_prefix_feedback只覆盖forward，checkpoint backward重算可能切回另一attention拓扑。022梯度协议PASS撤回，Judge needs_revision；数据/前向KV/执行事实保留。023编码可继续，训练必须先完成CPU梯度回归与第1次真实backward重算hook检查，optimizer前通过则自动继续128updates，失败硬停。外层context覆盖forward+backward；无新增GPU预算，补充01继续有效。[修复与门控任务](submission/experiments/EXP-023_v3_fm_small_data/ADDENDUM_02_CHECKPOINT_CONTEXT.md)。

2026-10-15 HKT。EXP-022工程训练通路已accepted；原Exp Worker立即执行32条真实视频的普通FM小数据拟合。CPU自检、编码pilot、批量编码、128updates及固定前后评测均已授权，正常检查通过即连续执行，不等额外marker。

[任务书](submission/experiments/EXP-023_v3_fm_small_data/taskbook_v1.md) · [授权](submission/experiments/EXP-023_v3_fm_small_data/AUTHORIZATION.json) · [固定选样](submission/experiments/EXP-023_v3_fm_small_data/SELECTION.json)。

- 28train+2dev新增编码，4train复用，先两clip pilot；不编码另6dev。
- 零增量新adapter初始化、普通FM C2训练，32样本×4epochs=128updates，每步重建GT C1 KV。
- 0/128updates：4固定clip GT-history、2dev self-generated-history；全部FM-8NFE，同条件前后评测，共12条视频。
- 上限13200GPU秒、440顶层DiT、128backward/updates、62VAE encode、12decode、600text；训练1卡，编码/评测最多2空闲卡。完整阶段限制/硬停见任务书。
- 不预设128updates收敛，不把profile容量当足够容量；无新增AnyFlow/DMD。

EXP-022：413.200GPU秒，11F/2B/2U，零adapter150项KV与velocity exact；allocated20.293GiB、采样点RSS最大64.704GiB。新增36RGB物化完成，画质/动作改善尚未测。[最终审核](submission/experiments/EXP-022_v3_fm_training_profile/judge/FINAL_REVIEW.md)。

Judge下发后立即消息通知固定Worker并确认开工，Worker完成后消息交回Judge验收与push。

发布与交接：EXP-022最终证据及EXP-023任务/授权已push main `7f9f327e17357c222523687cb15b3fc5eb014e02`，独立ls-remote与HEAD一致；固定Worker已确认EXP-023开工，按授权连续执行。

**EXP-023补充01（优先于上文旧预算）：** 已确认旧4train latent来自重编码MP4。本轮34clip统一source RGB重新编码I0/full17，原4条不再复用旧latent。VAE encode上限70、编码4200GPU秒、总13800GPU秒，其余128updates/评测/资源不变；已批准Worker直接执行。[补充任务书](submission/experiments/EXP-023_v3_fm_small_data/ADDENDUM_01_SOURCE_RGB.md)。

EXP-023补充01已通过会话消息发给原Worker，要求统一34clip source RGB输入；已push main `ba47b938cfbc1c09c690c3a43fdc9659a8035921`，独立ls-remote与HEAD一致。

EXP-022梯度验收更正与EXP-023补充02已push `53d21688a646ae20972ae091dd75f0edd3ba78c9`，远端SHA独立核对一致；修复执行消息已发给固定Worker，编码继续，optimizer前按新门控验证。

**EXP-023补充03：** 34clip编码完成，checkpoint0回归PASS；首次pre-eval在ffmpeg写片时失败（13DiT/1decode/301.064GPU秒），尚0训练。已授权显式ffmpeg路径+CPU codec检查及一次最小恢复，失败成本累计，decode上限13，其余440DiT/13800秒/128updates等不变。之后直接继续原任务。[恢复任务书](submission/experiments/EXP-023_v3_fm_small_data/ADDENDUM_03_VIDEO_IO_RECOVERY.md)。

EXP-023媒体恢复补充03已push `61be4c5d82200c7e43baa860f2b0111de484efd4`，独立远端SHA一致；已消息通知原Worker立即按授权恢复，pre/post各6片，失败费用累计，模型总预算不变。

**EXP-023补充04：** 第二次媒体故障（BF16.numpy identity）丢失首片，新增失败9DiT/1decode/81.937秒。Worker与Judge完整CPU保存/解码/故障恢复测试已通过，已授权attempt02最后一次首片恢复；总DiT449/decode14，13800GPU秒和128updates等不变。两次失败全部计入，之后媒体错误只从已保存RGB CPU恢复。[补充04](submission/experiments/EXP-023_v3_fm_small_data/ADDENDUM_04_BFLOAT16_RECOVERY.md)。

补充04/CPU媒体独立复核已push `d74d39d250ee102955d08e7fd1894429835ab9b1`，远端SHA一致；已消息通知固定Worker按最终首片恢复授权立即执行。

**EXP-023补充05：** 6条pre-eval已完成。训练首步hook安装早于context校验导致停止，0backward/0update，失败26.962秒/2DiT reserve保留。生产helper已由Worker和Judge独立CPU验证，现授权只恢复train→post，不重做数据/pre。总预算保持449DiT/14decode/128updates/13800秒，真实首步梯度门控通过后自动继续。[补充05](submission/experiments/EXP-023_v3_fm_small_data/ADDENDUM_05_TRAIN_HOOK_RECOVERY.md)。

EXP-023补充05/生产hook CPU复核已push `fd664dec23401711529c508364a10c1a0437fba4`，远端SHA一致；已向固定Worker发送立即恢复train→post指令，不重复已完成pre。
