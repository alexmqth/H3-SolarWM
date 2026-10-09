# VAE边界 / native time / global RoPE / future leakage

**状态：implementation verified; visual attribution limited**

12latent prefix前34RGB与完整decode一致，末5RGB可能回改。V3冻结已输出39帧再追加；action-video索引未发现切错，不以此排除所有表示问题。

## 原报告、数据、代码与视频

- [reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/vae_audit.json](../../../reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/vae_audit.json)
- [reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_audit/README.md](../../../reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_audit/README.md)
- [reports/stage1_anyflow/04_numerical_checks/README.md](../../../reports/stage1_anyflow/04_numerical_checks/README.md)

视频及逐运行指标沿用上述原报告内的真实链接。完整文件/视频索引：[整理前视频索引](../../../archive/reorganization_20261010/video_index.tsv)。

[返回本分支](../README.md)；本次没有新训练或模型推理。
