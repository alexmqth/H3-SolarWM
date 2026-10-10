# 核心实现阅读快照

这些文件供汇报与代码定位，不是可直接启动33B的独立runtime。完整运行仍依赖底座、released LoRA、DiffSynth和原始运行配置。

- [h3_cached.py](h3_cached.py)：来源 `code/causal/h3_cached.py`。
- [prototype.py](prototype.py)：来源 `code/causal/prototype.py`。
- [benchmark.py](benchmark.py)：来源 `code/causal/benchmark.py`。
- [diffsynth_causal.patch](diffsynth_causal.patch)：来源 `code/diffsynth_causal.patch`。

这是提交包commit 1a93713e218b8fd48d3fa5c47f25638068c012e0 的可追溯实现快照。V0/V1早期逐次运行未全部保存冻结runtime，不能把现行文件冒充当时逐字节源码；V2a主要接口也以原配置和checkpoint哈希为准。

[SOURCE_MANIFEST](SOURCE_MANIFEST.json)记录逐文件SHA256。[返回版本说明](../README.md)。
