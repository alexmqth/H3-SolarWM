# FM48固定权重的density对照：完整39帧

natural gt 30/chunk; joint keys/camera, not pure A/D

两自然场景来自held-out真实ABot。GT-history为oracle历史，不能当free rollout；自然片段之间flow差不能当A/D控制恢复。

时间为各入口记录的单次共享主机耗时，不能跨自然/停车场入口混用或宣称公平加速。GPU peak为入口记录的allocated峰值；没有独立分解weights/activations，CPU KV另列，不能将offload显存差归因为算法收益。MAD是帧差/活动量，不是画质；需检查全部39帧，尤其18–38帧重影和人物结构。

- [118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140_comparison.mp4](118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140_comparison.mp4)
- [dfec8ed3237860eba14d67c089ecd041_D_1750_comparison.mp4](dfec8ed3237860eba14d67c089ecd041_D_1750_comparison.mp4)

[逐条指标](metrics.csv) · [完整记录](metrics.json)

**自动产出不代表画质PASS；本报告不会替换meeting视频。**
