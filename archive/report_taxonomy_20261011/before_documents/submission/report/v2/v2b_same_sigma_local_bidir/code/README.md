# 核心实现阅读快照

这些文件供汇报与代码定位，不是可直接启动33B的独立runtime。完整运行仍依赖底座、released LoRA、DiffSynth和原始运行配置。

- [run.py](run.py)：来源 `reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/run.py`。
- [interval_forward.py](interval_forward.py)：来源 `reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/interval_forward.py`。
- [local_topology.py](local_topology.py)：来源 `reports/stage1_anyflow/02_causal_diagnostics/local_topology/runtime/code/causal/local_topology.py`。
- [h3_cached.py](h3_cached.py)：来源 `reports/stage1_anyflow/02_causal_diagnostics/local_topology/runtime/code/causal/h3_cached.py`。

V2b文件取自已冻结的实验runner/runtime，哈希可对原launch/runtime manifest核验。源码中的外部路径保持原样，不在汇报包里启动。

[SOURCE_MANIFEST](SOURCE_MANIFEST.json)记录逐文件SHA256。[返回版本说明](../README.md)。
