# EXP-022 验收更正01：checkpoint重算attention context

2026-10-15 HKT。**撤回原“工程训练梯度协议PASS”，Judge状态改为needs_revision。** 数据物化、checkpoint0前向/KV一致、调用和成本、旧状态保存重载一致的事实保留。旧两次更新的V3梯度正确性不被接受，数值影响未量化。原始runner/日志/checkpoint保留用于追溯，不修改或重跑旧实验。

原因：candidate内current_prefix_feedback仅覆盖forward；loss.backward发生在退出context后。non-reentrant checkpoint反向重算重新调用block，其causal_control.attend动态查找已恢复的类方法，存在prefix→current-video边变化。Judge此前核查了前向、finite梯度和更新计数，却漏查重算期间context生命周期；原验收过宽，在此更正。

修复/验证迁至EXP-023补充02：外层context覆盖完整forward/backward，CPU checkpoint与无checkpoint梯度回归及敏感负对照，真实首步重算hook确认，optimizer前门控。旧step02不用于后续初始化。无需为旧2updates重复GPU，仅使用新任务首步已有预算验证。
