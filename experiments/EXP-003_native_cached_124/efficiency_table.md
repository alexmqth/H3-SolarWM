# EXP-003 73→124 帧增量成本

同一台机器的 GPU 0，Original H3 + released action LoRA，30 native FM steps/chunk，832×480，seed 13，单路径一个进程续三个 5-latent chunk。下表 sampling 包含真实 CPU raw-KV 读取/搬运与计时事件开销；`GPU transfer` 是 sampling 的组成部分，**不能再加到 sampling 上**。每块新生成 17 RGB 帧，首 73 帧为 EXP-002 已保存结果，完整从零 E2E 未测。

| 路径 / 新增 RGB | Sampling 30次 | KV GPU transfer | Clean commit | Cache 文件保存 | VAE | 采样前 CPU KV | 块边界灰度 MAD | 水平 flow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AA 73–89 | 148.45 s | 19.47 s | 6.08 s | 12.52 s | 7.70 s | 12.465 GB | 13.82 | +1.463 |
| AA 90–106 | 176.43 s | 25.47 s | 7.00 s | 16.78 s | 9.86 s | 15.298 GB | 9.69 | +0.953 |
| AA 107–123 | 200.42 s | 34.97 s | 0 | 0 | 11.41 s | 18.131 GB | 2.91 | +1.335 |
| AD 73–89 | 148.94 s | 19.60 s | 6.08 s | 12.58 s | 7.70 s | 12.465 GB | 7.18 | −1.304 |
| AD 90–106 | 176.66 s | 25.56 s | 7.01 s | 16.85 s | 9.95 s | 15.298 GB | 3.34 | −0.648 |
| AD 107–123 | 199.43 s | 34.66 s | 0 | 0 | 11.40 s | 18.131 GB | 2.50 | −1.240 |

请以 [原始逐块 JSON](artifacts/metrics/) 的未四舍五入数值为准。AA/AD 两条进程总 wall 分别为 **798.95 / 773.32 s**，包含模型加载、已有 cache 读取、首个 clean commit、采样、缓存/endpoint 保存、VAE、视频编码及人工 review gate 等待，不是纯 sampling 耗时。任务总共 **1572.27 GPU-seconds = 0.43674 GPU-hours**，记录到的新块最大 `torch.cuda.max_memory_allocated` 为 **26,876.70 MiB**；首个第三块 commit 的峰值只执行了 44 GiB 上限断言，未持久化精确数值，不能把上述值称作完整进程峰值。每条最后一块没有多余 clean commit。

参考旧 V2b AA 同三个块的 sampling 是 **310.26 / 377.53 / 447.19 s**；本候选约为 **148.45 / 176.43 / 200.42 s**。这些是已记录条件下的观察成本：V2b 每步重算历史且采用 Same-σ 局部双向协议，本候选使用固定 clean raw KV、严格分块因果和计时事件。由于拓扑、历史条件、缓存与测量方式都不一致，不能把差值归因于单独的 KV cache，也不能将 Original 的整片 30-step 时间与本候选每块30-step的73→124帧增量时间直接相除称作整体加速。
