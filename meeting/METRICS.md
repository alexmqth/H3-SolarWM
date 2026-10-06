# Meeting metrics: current RGB visual-main

本表对应 124 帧 / 5.17 秒。新增 [243/481 帧长视频指标及视觉观察](long_horizon/README.md) 和 [CSV](long_horizon/METRICS.csv) 单独记录；同一 RGB checkpoint 的 20 秒 causal 视频严重崩坏，不能把本表的短片改善外推成长时稳定。

这些数据对应 `h3world_rgb_stable_*` 主视频：tail16 visual QKV adapter、RGB-consistent dual anchor、generated history、persistent CPU raw KV、8 steps/chunk。每个动作只有一次已完成记录，没有 warmup 或多次均值；因此只作协议内的工程测量，不能作严格 speed benchmark。

## End-to-end and memory

| Action | Original H3 30-step e2e (s) | Causal RGB-main e2e (s) | Original peak GPU (MiB) | Causal peak GPU (MiB) | Causal CPU raw KV (MiB) | Causal first chunk (s) | Causal mean chunk (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| W | 441.5 | 767.9 | 39925 | 31570 | 13509 | 57.4 | 88.6 |
| S | 444.8 | 673.4 | 39925 | 39940 | 13509 | 41.8 | 75.1 |
| A | 454.2 | 699.5 | 39925 | 39940 | 13509 | 43.1 | 78.9 |
| D | 450.4 | 715.9 | 39925 | 39940 | 13509 | 43.2 | 80.8 |

GPU MiB 转换后约为 causal 30.8–39.0 GiB；CPU raw KV 约 13.19 GiB。W 的显存峰值低于其他三次，来自单次运行的 offload/allocator 状态，不应解读为模型结构差异。

## Video continuity and action response

| Action | Original mean RGB MAD | Causal mean RGB MAD | Original boundary MAD | Causal boundary MAD | Original horizontal flow | Causal horizontal flow |
|---|---:|---:|---:|---:|---:|---:|
| W | 4.320 | 3.188 | 4.969 | 4.115 | -1.111 | -0.939 |
| S | 4.134 | 4.424 | 4.703 | 5.435 | -1.084 | -0.883 |
| A | 4.520 | 3.185 | 5.142 | 3.787 | +1.077 | -0.784 |
| D | 4.458 | 3.036 | 5.389 | 3.887 | -1.602 | -1.007 |

`MAD` 较低不等于画面更好，可能表示模糊或运动变弱。完整帧检查显示 RGB-main 比旧 fixed-mix 主视频更少出现人物透明/车库 tearing，但仍有 blur/ghosting。

A/D separation：

```text
Original H3:  +1.0767 - (-1.6019) = 2.6786
RGB-main:    -0.7841 - (-1.0075) = 0.2233
```

因此当前主结果没有通过严格动作 gate：`flow(A)>0`、`flow(D)<0`、`A-D>1.0`。A 的 image-space flow 甚至仍是负值；这不是完整 action preservation。W/S flow 在此停车场场景中不作为严格前后动作判定，只作辅助 proxy。

## Interpretation

1. Causal/KV 生命周期工作：124 帧、8 chunks、64 noisy forwards、8 clean commits，视频完整解码。
2. RGB-consistent anchor + visual adapter 改善了这组 124 帧样本的人物/场景结构，但这里同时改变了多项协议，不能归因于一个单独模块；20 秒长度测试未保持该效果。
3. 当前 causal-main 比原始 H3 更慢（673–768 s 对 442–454 s），所以不能宣传为端到端加速。它证明的是 causal execution / history reuse feasibility。
4. 视觉修复和动作几何是两个独立问题：视觉结构变完整，并不代表 A/D 方向恢复。
5. 早期 fixed-mix 的 A-D=0.453、383–452 s 表格已归档到 [`diagnostics/legacy_fixed_mix/`](diagnostics/legacy_fixed_mix/)，不能与本表混用。

原始 JSON、run metadata、flow 和 continuity 数据见 [`source_metrics/rgb_visual/`](source_metrics/rgb_visual/)。
