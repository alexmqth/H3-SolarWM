最新完整64次结果：[FINAL_RESULTS.md](FINAL_RESULTS.md)。下文保留当时快照。

# Full-history / train shift12：32次更新训练结果

记录时间：2026-10-08T11:29:07.255211+08:00。新增16次训练完成，A/D39f的4/8步评测全部结束并失败，见[STEP32_RESULTS.md](STEP32_RESULTS.md)。

本段含准备/验证耗时3012.30秒，allocated峰值40667.85MiB。恢复来自本支自己的step16/Adam/RNG。旧visual/time冻结，四类LoRA均有更新；训练分布shift12，公共验证/推理2.22。

| Action | 更新数 | Weighted total | Diffusion 1 | Diffusion 2 | Endpoint raw | Flow-map raw |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 0.118051555 | 0.064436495 | 0.140979081 | 59.624652863 | 0.168970197 |
| A | 16 | 0.118034674 | 0.064428739 | 0.140957251 | 54.556304932 | 0.170321628 |
| A | 32 | 0.117911468 | 0.064365514 | 0.140805602 | 59.536766052 | 0.170173332 |
| D | 0 | 0.170266438 | 0.096876085 | 0.198909059 | 22.804769516 | 0.228104293 |
| D | 16 | 0.170265539 | 0.096842855 | 0.198944777 | 24.329162598 | 0.226440981 |
| D | 32 | 0.170028823 | 0.096661933 | 0.198720217 | 22.905317307 | 0.224508852 |

从16到32，A endpoint上升（54.556→59.537），D下降（24.329→22.905），其它变化较小。与shift2.22分支相反；这还不能判断真实视频效果。不能用weighted total代替原始residual或动作/画质验收。

三时点公共sigma/r一致；恢复前后validation逐项相等；本次全部32个更新的action/chunk和128个sigma/r与事先CPU计划完全相同。该检查不包含actual GPU noise hash，不能把预测noise hash当实际采样审计。

GPU1原队列评测A/D8-step；GPU4只读同一step32评测A/D4-step，补齐与匹配FM32的步数对照。数字gate不等视觉通过，两组均未过动作和视觉联合gate。

冻结benchmark日志anyflow_training_shift误标为2.22；真实训练shift12以training.json time_sampling、checkpoint/setup config为准。原始证据不改写。
