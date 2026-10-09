# C_anyflow_dmd

按研究问题归档；下表区分实现、训练完成和能力验收。原报告/原始日志保留原路径与字节。

| 子实验 | 状态 | 结论 |
|---|---|---|
| [TF-AnyFlow实现与数值验证](01_tf_anyflow/README.md) | implementation verified; full Stage1 not accepted | 有限区间finite-map、目标时间条件和r=t/符号/梯度等检查存在；ordinary FM与AnyFlow目标明确区分。 |
| [AnyFlow16 vs matched FM16](02_anyflow16/README.md) | preliminary trained; quality failed | 两条Original A/D伪标签；16更新；4/8steps/chunk。训练初始化来自旧RGB causal协议，不是V3。等更新不等算力。 |
| [AnyFlow64/128/136 budgets](03_anyflow64_128_136/README.md) | preliminary trained; quality failed | 这些预算实际存在。128的8步A−D0.7597，136两臂0.5810/0.6352；4步后段雾化，内部一致性下降不构成成功。 |
| [Finite-interval / diagonal / teacher endpoint](04_finite_numerics/README.md) | implementation verified; ability not established | 同时保留finite-vs-diagonal自一致性和对Original teacher endpoint的误差；两者都不单独等于视觉质量。 |
| [Real ABot与AnyFlow/FM数据划分](05_real_data_objectives/README.md) | objective audited; real ABot AnyFlow run not found | 目录名stage1_anyflow不代表全是AnyFlow。已查训练：AnyFlow用Original伪标签，真实ABot主要FM48/E2；未找到真实ABot AnyFlow训练结果，不能编造。 |
| [Trainable fake score / self-rollout / DMD surrogate](06_stage2_lite/README.md) | implementation verified; preliminary trained; quality failed | 共享H3的student/teacher/critic及实际小预算DMD链路存在，旧latent版人物分解，RGB集成A/D仍未恢复。不称完整SolarWM Stage2成功。 |
| [Shared roles / FMBS / DMD gradients](07_dmd_preparation/README.md) | implementation verified on small tests; 33B quality not validated | CPU/小H3梯度、角色隔离和FMBS工程测试不升级为完整33B Stage2。V3还未接AnyFlow或DMD。 |

[返回研究分支总览](../README.md) · [主线版本](../../mainline/README.md) · [精简汇报](../../report/README.md)
