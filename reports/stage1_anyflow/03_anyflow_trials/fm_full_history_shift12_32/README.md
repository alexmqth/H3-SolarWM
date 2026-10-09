# Full-history/shift12/FM32匹配对照

控制器515452已启动，GPU3/reserve6。固定相同初始化、full-history、rank8、数据/time/noise与32次更新预算，比较普通FM与AnyFlow目标/采样。32次训练已完成且起点/完整采样序列审计通过，A/D4/8视频正在生成；[训练结果与实际前向计数](GPU_RESULTS.md)。尚不宣称任何方法有效。

[计划](PLAN.md)、[启动检查](preflight.json)。controller依赖源机器冻结runtime，通用复现入口见REPRODUCE.md。

真实GPU初始三套adapter与AnyFlow完全一致，CPU/CUDA/logical RNG也相同；公共validation每动作前两个r=t样本的raw loss逐项相等。见[gpu_initial_audit_early.json](gpu_initial_audit_early.json)。这仅确认匹配起点，不能证明训练或视频有效。
