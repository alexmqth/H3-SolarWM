# Checkpoints

[因果基线配置与视频](../docs/CAUSAL_BASELINE.md)及[逐文件固定清单](../reports/stage1_anyflow/07_protocols/causal_baseline.json)区分已有adapter与零训练局部协议。没有新增或宣称合格的causal权重。

这里仅放小型实验 adapter，不放 MiniMax-H3 33B backbone、不放 H3-World 发布的基础 LoRA，也不放数据集或 cache。

- visual_rgb_tail16/causal_adapter.pt：RGB-consistent visual tail16 causal adapter。
- visual_rgb_tail16/action_adapter.pt：对应 fixed-mix/action residual。
- stage2_lite/critic_action_adapter.pt：Stage2-lite fake-score critic adapter。
- action_diagnostic/student_action_adapter.pt（也是 Stage2-lite 的 student，二者字节相同）：corrected own-history endpoint + paired-QKV 诊断 adapter；它没有通过 action gate，只用于复现实验负结果。

adapter 必须和同一版本的 H3-World LoRA、DiffSynth patch、anchor protocol 一起使用。不要把这些 adapter 单独解释为完整 causal checkpoint。
- legacy_fixed_mix/action_adapter.pt：保留较强 A/D 符号响应、但后段视觉漂移的旧 checkpoint；搭配 latent dual、own prefix、feedback off。具体复现见该目录 README。
