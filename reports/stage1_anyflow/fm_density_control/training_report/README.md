# FM密度对照：48次训练完成，效果验收待评测

从同一初始化完成两个48-update分支，复用已完成shift12对照。新分支只改变数学协议中的训练sigma采样shift为2.22，Gaussian权重函数保持shift12；采样变化会改变实际scalar weight和gradient mass，没有importance correction。

| 验证noise段 | 数量 | 初始化raw MSE | FM48 shift12 | FM48 shift2.22 | 新分支相对旧分支 |
|---|---:|---:|---:|---:|---:|
| low | 0 | 未测 | 未测 | 未测 | 不推断 |
| mid | 12 | 0.181478 | 0.178895 | 0.177215 | -0.94% |
| high | 4 | 0.086299 | 0.081888 | 0.082432 | +0.67% |
| all | 16 | 0.157683 | 0.154643 | 0.153519 | -0.73% |

负百分比表示raw residual更小。中段有所改善，高段均值略差，不能称为全面改善；all均值由本组12个mid/4个high的固定数量决定，不是自然噪声分布下的期望性能。
完整16项初始validation记录两分支相同，clip/chunk/sigma/weight及验证策略匹配。全部为最后chunk2（两latent帧），不能代替所有chunk的检查。历史control未保存validation完整noise tensor哈希，证据限于源码、种子与记录匹配。这里是GT clean-history局部FM拟合，不是当前A/D反事实或视频验收。

| 运行 | 记录训练wall(s) | allocated peak(MiB) | GPU | reserve(GiB) |
|---|---:|---:|---|---:|
| shift12 | 10380.96 | 30125.81 | 0 | 14.0 |
| shift2.22 | 6656.35 | 34683.62 | 1 | 14.0 |

两次在不同GPU和共享负载下运行，allocated峰值也不同；不将训练wall差归因于采样分布或宣称算法加速。训练峰值不是推理峰值，GPU权重/激活未独立分解。
最终checkpoint/source/课程检查通过；另有step48 Adam/完整noise RNG审计。预算已封顶，没有追加更新。后续54点noise扫描、GT/generated各18点A/D及10条39帧视频由原队列继续执行；完整结果和全帧评审之前不通过B/C/D质量门槛。

[逐项CSV](metrics.csv) · [完整比较](metrics.json)
