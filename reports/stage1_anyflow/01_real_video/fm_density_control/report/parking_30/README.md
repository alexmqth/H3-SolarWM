# FM48固定权重的density对照：完整39帧

parking current A/D, 30/chunk

Original是旧legacy精度/原始条件，两个FM48采用同h3_fp32 causal/RGB协议；与Original为pipeline参照，两种density才是受控训练比较。

时间为各入口记录的单次共享主机耗时，不能跨自然/停车场入口混用或宣称公平加速。GPU peak为入口记录的allocated峰值；没有独立分解weights/activations，CPU KV另列，不能将offload显存差归因为算法收益。MAD是帧差/活动量，不是画质；需检查全部39帧，尤其18–38帧重影和人物结构。

- [A_comparison.mp4](A_comparison.mp4)
- [D_comparison.mp4](D_comparison.mp4)

| 方法 | A flow | D flow | A−D | 仅数值门槛 |
|---|---:|---:|---:|---|
| Original30 full-seq | +1.181253 | -0.842124 | 2.023377 | True |
| FM48 shift12 30/chunk | -0.182680 | -0.163995 | -0.018685 | False |
| FM48 shift2.22 30/chunk | -0.190239 | -0.158556 | -0.031684 | False |

[逐条指标](metrics.csv) · [完整记录](metrics.json)

**自动产出不代表画质PASS；本报告不会替换meeting视频。**
