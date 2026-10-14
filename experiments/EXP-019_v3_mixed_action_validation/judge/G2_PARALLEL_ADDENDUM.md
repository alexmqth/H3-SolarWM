# EXP-019 G2 并行调度补充授权

2026-10-15 03:06 HKT，Judge。用户明确要求利用两张额外空闲卡并行缩短等待时间。本补充优先于taskbook_v1、G1_REVIEW_AND_G2_RELEASE以及旧marker中的“单卡串行/固定执行顺序”调度限制；原冻结模型代码/来源清单/输入/scene集合/实验次数不变。

## 分配与执行者

- 已运行7199292c保持原GPU0/原进程，不重启、不重复。
- 9dc2e588可由既有Exp Worker在GPU2启动。
- b784d995可由既有Exp Worker在GPU3同时启动。
- 启动前必须重查实际空闲状态；GPU1与4–7当前有其他任务，不使用/不操作。若指定卡已被占用，Worker报告并等待空闲，不抢占。
- 项目最多同时3卡，09:00后也满足≤3卡；无需新建agent。并行由同一Exp Worker管理独立进程与报告。

## 不变预算与证据

每scene39forward（1commit+30+8）、2decode、900秒；G2总117forward/6decode/2700GPU秒，完整EXP-019总166forward/14text/8decode/5400GPU秒。GPU秒为各任务实际单卡wall相加，不用并行总经过时间替代。不新增scene/seed/NFE/训练，不因更快完成而追加实验。

每进程独立CUDA_VISIBLE_DEVICES、输出目录、账本和watchdog；记录GPU UUID、启动时间及CPU affinity。可选取各自不同的4CPU集合，避免多个进程共用同四核；当前系统available内存约504GiB，仍监测CPU offload/KV内存竞争。共享模型文件只读，不改冻结run.py。并行可能有I/O/CPU竞争，成本对比明确披露，不能把单次wall差异全归因算法。

## 停止条件

NaN/OOM/超时停止对应任务并报告；发现共用协议/输入/实现错误时停止所有受影响任务。某scene视觉失败不自动否定其他独立scene，已启动的其余有界任务可完成并记录负结果；失败scene不重试或扩展。若尚未启动则先报Judge决定。此条替代此前串行“失败后不启动后续”的调度依赖，保留ROI与总预算。

Worker负责执行、实物与报告；Judge负责审核、next_plan/progress和Git发布。
