# Next Plan v5：冻结研究实验，完成面试交付

2026-10-09按用户指定的四项优先级收尾。E2局部动作＋结构验收仍为No-Go；不因工程复现通过而改变该结论。

| 优先级 | 任务 | 当前结果 |
|---|---|---|
| 1 | 收齐E2并冻结 | FM-only/FM+action各4更新，A-history、D-history和两条真实GT-history全部收齐；无一致改善，记录负结果。16份关键证据哈希及14个历史进程身份已核查，见[冻结收据](../../reports/stage1_anyflow/01_real_video/real_transition_windows/FREEZE.json) |
| 2 | 更新提交包和会议材料 | INTERVIEW_ANSWER、EXPERIMENT_REPORT、REPRODUCE及meeting已统一；明确旧主片checkpoint与AnyFlow/E2/DMD准备的界限 |
| 3 | 最终可复现性验收 | 新venv＋GitHub新DiffSynth源码；15项KV/局部因果测试通过；随机小H3训练smoke通过；真实33B旧adapter生成39f并完整解码，测量和哈希齐全 |
| 4 | 5分钟技术答辩 | 六页提纲和含播放时间讲稿完成；Original vs causal开场，完整20秒失败片和E2负结果作诊断证据 |

本轮没有把E2从4扩到16，没有重启AnyFlow或Stage2。验收训练仅为随机小H3固定合成batch的8次smoke更新，不是研究模型训练。当前GPU共享负载，不新增不公平的124f性能排名，保留历史单次表并明确限制。

入口：[面试回答](../../INTERVIEW_ANSWER.md) · [实验报告](../EXPERIMENT_REPORT.md) · [复现](../../REPRODUCE.md) · [验收证据](../../reports/final_acceptance/README.md) · [会议讲稿](../../meeting/MEETING_SCRIPT.md)。

## 后续研究，仅保留建议，不自动执行

可信局部causal30step → AnyFlow4/8step → on-policy Stage2。先验证局部动作信息流、action-dependent prefix重算、监督覆盖及动作后果时序；不要求先解决全部长时漂移，但也不以Stage2解释全部GT/reference历史下的局部结构问题。

当前交付状态：工程实现与复现可行；动作保真、稳定长视频和端到端加速没有同时成立。完整历史计划和实验过程保留在[PROJECT_PROGRESS](PROJECT_PROGRESS.md)与各原始报告中。
