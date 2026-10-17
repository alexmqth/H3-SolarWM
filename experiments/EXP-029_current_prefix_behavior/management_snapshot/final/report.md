# EXP-028/v1 Worker report

2026-10-17 HKT. **已完成获批的有限 GPU 诊断；机制结论未成立，等待 Judge 验收。** 两条真实 ABot 片 × 两个 sigma × Original-visible/V3 共 8 个 factual C2 FM 图，在 H3 released Action LoRA 上测量固定 rank-8 LoRA-B 的 action/video/prefix 角色梯度（层 0、49）。恢复运行完成 14 次顶层 DiT、8 次 backward、0 更新、0 视频；4 次零 B 输出恒等对照逐值相等，16 个角色和相对误差最大 6.701×10⁻⁷。峰值 allocated/reserved 显存 23.753/26.770 GiB。恢复进程从 scheduler 启动到退出为 **580.532 秒**；进程内账本为 572.230 秒，少计启动/退出的 8.302 秒。首个导入失败进程精确 3.268 秒，按授权保守计 4 秒；两次实际进程时间累计 **583.800 秒**，保守预算口径 **584.532 秒**。GPU1 与他人进程共享，耗时不能当独占吞吐。

主要观察：L0 的 action–video LoRA-B cosine 在两个图中均接近零；L49 V 为 Original −0.1075、V3 −0.1119，未见 V3 一致且明显更负的冲突。V3 历史 writer 梯度为结构零，原因是 GT C1 KV detach，并不表示模型不读取历史。此实验没有动作反事实视频，不能以 cosine 或 FM loss 判定动作控制/视觉质量，也不足以支持双 Adapter 训练。精确角色 dot 和跨图梯度矩阵没有持久化，明确列为缺失项。只建议下一轮做同 C1/KV、同 C2 噪声的 A/D 短片，在 V3 与仅关闭当前非动作 prefix→video feedback 之间进行单边行为对照；本轮未启动。

完整协议、首次启动失败、CPU 门控、8 条逐图结果、成本和后续建议见[EXP-028 Worker 报告](submission/experiments/EXP-028_action_video_role_probe/report.md)。Worker 未操作 Git、`progress.md`、`next_plan.md` 或冻结源码。
