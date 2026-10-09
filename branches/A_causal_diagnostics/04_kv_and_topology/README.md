# Recompute / Persistent KV / Strict video causalization

**状态：implementation verified under controlled numerics**

统一SDPA可见组与GEMM形状后，6状态×A/D的50层K/V、RoPE和velocity误差0。无persistent KV的R2对Original动作差分cos已0.057；生产默认不声称bitwise等价。

## 原报告、数据、代码与视频

- [reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md](../../../reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md)
- [reports/stage1_anyflow/02_causal_diagnostics/local_topology/README.md](../../../reports/stage1_anyflow/02_causal_diagnostics/local_topology/README.md)

视频及逐运行指标沿用上述原报告内的真实链接。完整文件/视频索引：[整理前视频索引](../../../archive/reorganization_20261010/video_index.tsv)。

[返回本分支](../README.md)；本次没有新训练或模型推理。
