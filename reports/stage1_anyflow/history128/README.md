# AnyFlow128：teacher/generated history受控诊断

同一shift12 step128 checkpoint，39 RGB/12latent/3chunks5+5+2、8 steps/chunk、RGB dual、CPU raw KV、causal action rows与feedback、seed13、inference shift2.22均保持。只将clean commit与下一块RGB anchor的历史换为对应Original H3 teacher latent。复用原冻结runtime，manifest增加128权重、训练/恢复收据和本控制器。

A在GPU5、D在GPU6独立运行，启动前检查显存；不碰GPU4原generated-history队列，也不启动训练。各run_A.json/run_D.json为实况。两条都完成并与generated对照后，才能形成结论。

本对照是oracle诊断，不是free-running结果或动作验收。历史包含动作后的状态；第一块latent应与对应generated结果相同，需逐张量检查。后续正flow不能独立证明当前action binding。原始帧全部保留，检查边界重置，不用单一全片flow掩盖不连续。
