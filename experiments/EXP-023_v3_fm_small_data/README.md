# EXP-023：V3普通FM小数据拟合

已授权Exp Worker执行32train/128updates、固定GT与self-generated-history前后对照，最多2卡用于编码/独立评测，训练1卡。FM为Flow Matching；128updates是训练更新数，FM-8NFE为每chunk8次推理求值。

[任务书](taskbook_v1.md) · [授权](AUTHORIZATION.json) · [固定选样](SELECTION.json)。任务尚待实际执行结果，不能称画质改善。

## 执行补充（优先于原任务书相应条款）

- [补充01：34clip统一源RGB，70次VAE编码/13800GPU秒](ADDENDUM_01_SOURCE_RGB.md)。
- [补充02：forward与checkpoint backward保持相同context](ADDENDUM_02_CHECKPOINT_CONTEXT.md)。编码继续；optimizer更新需先通过CPU梯度回归及首个真实backward重算门控，之后自动继续。旧EXP-022梯度PASS已撤回，前向/KV证据保留。
