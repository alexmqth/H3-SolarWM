# 同协议续训队列

[执行计划](PLAN.md)与[CPU预检查](preflight.json)。`run_extension.py`是源实验控制器归档，使用该实验的隔离runtime目录；迁移机器后应按STAGE1_ANYFLOW.md中的通用训练入口复现，而不是直接运行此归档脚本。

队列已启动等待冻结时间MLP16-update训练和全部A/D评测。当前没有续训结果，不把排队状态写成64-update已完成。对应源目录：H3-World/outputs/2026-10-08-04/stage1_anyflow39_frozen_extend64/。
