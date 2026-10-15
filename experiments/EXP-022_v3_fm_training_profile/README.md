# EXP-022：普通FM真实训练profile

状态：CPU实现及有限GPU执行已授权，交Exp Worker立即开始。目标为验证真实V3可微训练与资源成本，最多2次optimizer updates，不是2步推理，也不代表充分训练。

[任务书](taskbook_v1.md)。本轮不生成视频；评估产物是前向等价、梯度、checkpoint与成本记录。
