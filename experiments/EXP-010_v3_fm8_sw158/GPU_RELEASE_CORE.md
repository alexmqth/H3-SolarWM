# EXP-010/v1 Judge到124帧阶段放行

2026-10-11 05:09 HKT。Judge独立CPU重跑通过，36项冻结来源、旧FM8 AA73端点/自有KV、long47 0:37原始条件与噪声、12/17/22/27/32/37裁剪相等、C7/C8分叉、native8 grid与精确缓存祖先检查通过。CPU阶段发现并修复rollout_contract缺导入路径，补实际run依赖导入后重冻结；没有失败GPU消耗。

**现在批准实际空闲GPU0顺序执行commit_c3→C4→C5→C6。** 本阶段最多28forward=24sampling+4clean commit，3VAE，0训练；总任务上限62forward/7VAE/.75GPUh仍保留，单卡allocated≤44GiB，最晚09:00。每块保留原始日志、端点/RGB摘要与指标；C6提交真实淘汰C1，50层精确indices应为1–5，历史KV为14,164,800,000 bytes。

若人物/场景持续崩溃、nonfinite/OOM、来源/位置/缓存协议不一致则停止后续启动，普通残影/边界缺陷允许PARTIAL。无自动重试。完成C6后停在124帧供Judge看全部51新增帧；C7/C8无当前GPU授权，另需CORE_JUDGE_REVIEW.json与GPU_AUTHORIZATION_BRANCH.json。源码/config/输入绑定manifest c5be2dd41bb6b755f84ea5dea7580c0734f803177e8f57217a34380a666d41b1，不在运行期修改冻结依赖。
