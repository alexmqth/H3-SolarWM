# A_causal_diagnostics

按研究问题归档；下表区分实现、训练完成和能力验收。原报告/原始日志保留原路径与字节。

| 子实验 | 状态 | 结论 |
|---|---|---|
| [Chunk 5/7/12 与原生时间网格](01_chunk_partition/README.md) | implementation verified; V2b local PASS only | B7首窗A方向失败；C12→5 + N在第二块局部通过。均不能推广到124f。 |
| [Clean/Same-σ 与 Single I0 / latent / RGB](02_history_anchor/README.md) | mixed; protocol-dependent | 恢复原生条件取得首窗正控；Same-σ对历史续写有效，但更早12→12仍有ghosting。RGB联合修复不能当anchor单因素结论。 |
| [Single-Egress / feedback / public prefix](03_action_routing/README.md) | implementation verified; action mismatch diagnosed | causal action prefix允许当前V直接读past A；raw cache仅存video K/V，额外action直读来自fresh prefix。own仍允许通过历史video间接传播。 |
| [Recompute / Persistent KV / Strict video causalization](04_kv_and_topology/README.md) | implementation verified under controlled numerics | 统一SDPA可见组与GEMM形状后，6状态×A/D的50层K/V、RoPE和velocity误差0。无persistent KV的R2对Original动作差分cos已0.057；生产默认不声称bitwise等价。 |
| [VAE边界 / native time / global RoPE / future leakage](05_vae_time_rope/README.md) | implementation verified; visual attribution limited | 12latent prefix前34RGB与完整decode一致，末5RGB可能回改。V2b冻结已输出39帧再追加；action-video索引未发现切错，不以此排除所有表示问题。 |

[返回研究分支总览](../README.md) · [主线版本](../../mainline/README.md) · [精简汇报](../../report/README.md)
