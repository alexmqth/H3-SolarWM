# Single-Egress / feedback / public prefix

**状态：implementation verified; action mismatch diagnosed**

causal action prefix允许当前V直接读past A；raw cache仅存video K/V，额外action直读来自fresh prefix。own仍允许通过历史video间接传播。

## 原报告、数据、代码与视频

- [reports/stage1_anyflow/02_causal_diagnostics/first_chunk_routes/README.md](../../../reports/stage1_anyflow/02_causal_diagnostics/first_chunk_routes/README.md)
- [reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md](../../../reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md)
- [reports/action_alignment/ACTION_ROUTING_PROBE.md](../../../reports/action_alignment/ACTION_ROUTING_PROBE.md)

视频及逐运行指标沿用上述原报告内的真实链接。完整文件/视频索引：[整理前视频索引](../../../archive/reorganization_20261010/video_index.tsv)。

[返回本分支](../README.md)；本次没有新训练或模型推理。
