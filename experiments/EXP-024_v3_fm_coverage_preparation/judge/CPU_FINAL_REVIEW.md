# EXP-024 CPU阶段 Judge 审核与GPU放行

2026-10-15。CPU准备accepted；全层rank8/C1+C2的真实33B训练成本尚未测。参数量独立算得10,752,000；rank32为43,008,000，本轮只选择rank8。已审阅训练入口、50层tiny checkpoint、真实条件和恢复测试。GPU不以CPU tiny成功代替验证。

初版cache仅检查nbytes、路由完成后才验收；已要求修正。修正后每次optimizer前检查路由和冻结梯度，cache使用结构签名+完整K/V/RoPE字节摘要。Judge独立CPU故障注入通过：同容量第20元素KV改动、缺layer49、冻结参数有梯度均拒绝；前两者optimizer调用数为0，C1失败未进C2。见GATE_REVIEW.json。完整摘要成本计入profile，不追溯更改EXP-023签名检查的范围。

批准taskbook_v2_gpu_profile.md的单卡最多3顶层DiT/2backward/2updates/1800秒。Worker先完成源码清单冻结与既有CPU回归，然后立即启动，无需等待额外报告marker。失败不自动重试，不追加训练或视频。此次为工程可执行性与资源profile，不作质量结论。
