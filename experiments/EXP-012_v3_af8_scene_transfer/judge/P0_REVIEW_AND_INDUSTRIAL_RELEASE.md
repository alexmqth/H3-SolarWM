# EXP-012 P0 Judge审核与工业场景放行

2026-10-11 06:55 HKT。独立复跑preflight.audit与Worker冻结CPU审计逐值一致，预算封顶/finite-map符号/无marker拒绝测试PASS，0 GPU调用。人工核对实际runner在AF权重安装后自身重建C1 KV，clean source/target=0，native8相邻target、AA/AD仅C2动作行变更，immutable旧39RGB和单场景17forward/2decode成立。补齐加载/保存计时与commit显存检查；SIGALRM和逐调用预算/09:00闸门存在。

11项任务代码/config与34项来源清单已核对。额外记录interval_student依赖的chunk_plan/position_sw摘要于P0_REVIEW.json。显式模型根与冻结runtime根resolve到同一目录；没有模型来源或协议改变。真实权重配对、C1来源/噪声/位置与原FM8基线匹配；CPU结果不代表视频质量。

**批准GPU0工业场景一次运行：1clean commit+16sampling=17forward，2decode，0encode/update；总任务≤1260GPU秒、allocated≤44GiB、磁盘≥60GiB。** 启动前再次确认GPU0空闲；GPU3/4为他人进程，禁止操作。完成后检查全部AA/AD34新帧及实际cache/旧RGB；村落仍待独立放行。无自动重试、无C3、无调参。

输入与代码摘要绑定见G1_industrial_APPROVED.json。用户已授权夜间持续研究，放行由Judge在既定任务预算内执行。
