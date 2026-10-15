# EXP-021/v1 Judge验收

2026-10-15 HKT：accepted，CPU准备任务收口。Judge阅读来源/训练计划/实际代码，重新运行4项CPU单测通过，独立调用manifest审计检查40窗口、episode隔离与非重叠，并重算上游来源manifest SHA通过。

接受32train+8dev候选数据清单和普通FM接口草案。仅4train已物化，其余36窗口待抽帧；仅4train/2dev episodes，dev曾被观察，不是盲测。微型CPU测试不证明33B可微反向、真实cache/未来隔离或生成能力。真实训练入口尚缺，训练吞吐/显存NOT_TESTED，零GPU/零更新。tail8 QKV rank8只是profile用最小配置，不据其结果评价足够训练后的能力或认定参数量足够。

下一任务EXP-022实际实现可微入口与最多2updates profile，同时物化候选RGB。用户明确要求下发即通知Worker执行，本次CPU与有限GPU预算一起授权，满足任务书的自检条件即可连续执行，不再等待额外常规marker。任何协议差异/OOM/非有限值按任务书停止并报证据。
