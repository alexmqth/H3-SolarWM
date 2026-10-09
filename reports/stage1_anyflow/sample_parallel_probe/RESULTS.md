# 同一logical batch的四卡并行：真实33B探针通过

不改变架构、数据、噪声、global logical batch4、loss比例/缩放、历史梯度和Adam定义；每卡负责四个样本中的一个，四卡各自复制冻结H3。仍是原始每样本四次forward，没有叠加diagonal shortcut。这是样本并行，不是SP/模型分片。

真实shift12 step32、A/chunk2：四卡样本raw loss/weight/adaptive scale/weighted loss与同状态串行逐项相同；实际noise hash、logical/CPU/CUDA RNG相同；所有bank和冻结visual/time/action均未变。没有optimizer update或保存候选模型。

| 执行 | 批次耗时 s | GPU allocated peak MiB |
|---|---:|---:|
| Parallel rank0 | 82.267 | 40810.913 |
| Parallel rank1 | 82.264 | 40810.913 |
| Parallel rank2 | 82.264 | 40810.913 |
| Parallel rank3 | 82.267 | 40810.913 |
| Serial rank0 | 225.976 | 40973.210 |

四卡归约后梯度hash逐bit相同。相对串行：gradient cosine=0.999970379，difference norm=0.000173769208，max abs=3.98661359e-06。差异接近此前同模型重复串行反传的尺度（norm约1.58e−4），不声称CUDA逐bit串行等价。

单批四卡最慢rank为82.27秒，串行为225.98秒，观察到约2.75× wall比。四卡与一张卡资源不同；这是共享主机、一次顺序测量，排除模型准备，不是完整训练吞吐或推理speedup。物理clean commit从串行2次变成四卡合计8次；current16、带图history8保持global计数。

CPU四进程Gloo六个真实小H3 cases验证了逐样本loss和噪声、梯度与一次Adam更新，覆盖FP32/BF16和三个chunk；另外隔离trainer完成串行4、并行4、并行2+恢复4，以及旧串行step2→并行4。并行/串行bank最大差7.45e−9；并行2+恢复4与连续4的adapters、Adam和RNG逐bit相同；四replica参数/Adam/RNG都相同。详见cpu_training_equivalence.json。

带rank0唯一写checkpoint的训练入口已在training_runtime中隔离准备，只在CPU完成多更新/恢复检查；尚未启动真实33B多卡optimizer训练。主trainer和GPU0/1冻结runtime均未修改。当前仍先等待两组原64次训练的视频结果，Stage1画质/action gate未通过，Stage2暂缓。

四卡资源代价：每rank估算raw CPU KV峰值为[10806.884765625, 10806.884765625, 10806.884765625, 10806.884765625] MiB，四rank峰值之和约42.21GiB（是缓存张量字节估计，不是全进程CPU RSS）；四卡各有一份H3权重。多卡加速使用了更多总GPU/主机内存，不是单卡内存优化。
