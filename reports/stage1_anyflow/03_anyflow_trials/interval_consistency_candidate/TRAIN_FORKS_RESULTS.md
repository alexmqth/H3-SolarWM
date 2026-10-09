# 同128初始化的两组136训练均完成

两组各新增8个optimizer updates，未超过136。权重/Adam/RNG恢复审计、实际129–136动作/chunk/时间对核查，以及step128/132/136逻辑RNG核查全部通过。未直接记录GPU噪声tensor hash，不能冒充直接GPU逐张量核验。

| variant | new updates | update wall s | including setup/validation s | peak allocated MiB | current forwards | physical clean | differentiable history |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 8 | 1512.06 | 1888.10 | 40666.14 | 128 | 8 | 32 |
| auxiliary | 8 | 1510.40 | 1889.25 | 40667.62 | 136 | 8 | 40 |

计数不含validation和activation-checkpoint反向重算。辅助比control多8次current与8次history forward；等更新不等算力。共享CPU/GPU运行环境不同，wall秒数几乎相同也不能解释为额外目标免费。visual/action/time冻结、四类bank更新、bank容量/结构不变，见两组136_parameter_audit.json。

| variant | action | fixed endpoint raw residual before | after | frozen reference MSE before | after |
|---|---|---:|---:|---:|---:|
| control | A | 34.422421 | 22.487106 | — | — |
| control | D | 15.185422 | 14.106135 | — | — |
| auxiliary | A | 34.422421 | 30.798725 | 0.05760011821985245 | 0.050168395042419434 |
| auxiliary | D | 15.185422 | 14.712427 | 0.05909423902630806 | 0.05232534557580948 |

这里endpoint列是AnyFlow固定验证的raw PDE residual，**不是**finite endpoint到teacher clean的RMSE；后者在双局部指标报告中单独测量。辅助fixed-reference列也不是当前finite/当前diagonal自一致性，且是训练所用固定参考状态，不能冒充独立泛化测试。

训练与数值审计完成不代表Stage1视频通过；control4/8已失败，auxiliary完整4/8 A/D视频正在生成。见[完整视频与最终指标入口](README.md)。Stage2没有启动。
