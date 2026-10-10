# EXP-004：V3 Baseline 原权重 8-step native FM 续写

**当前研究分类：V3 Baseline Strict Causal + Persistent KV。** 原Efficient Causal路线曾称V3；历史任务/报告/指标/源码保留原称，现行分类见[V2家族](../../report/v2/README.md)。

**Worker completed；Judge accepted（有限可行性，画质/严格连续性 PARTIAL）。** 这不是 AnyFlow 训练或完整 8-step 视频：首 12 latent / 39 RGB 帧和首窗 raw KV 均复用 EXP-002 的 **30-step** 结果，只有新增的两个 5-latent chunk 各用 8 次原生 FM/Euler 更新。Original H3 + released action LoRA、Single I0、current-prefix own-action、严格 chunk-causal、CPU persistent raw KV、native timestep / global RoPE、sigma=0 clean commit、seed 13、初始噪声、音频条件和 flow shift 2.22 均未改变。

AA、AD 第二块从同一个 A 历史和 noisy state 出发，只改当前动作；第三块分别接自己的 8-step 第二块。两条路径均生成 56、73 RGB 帧，人物和停车场在新增帧中保持可辨，动作水平光流方向相反。8-step AA 第二块动作幅度小于 30-step；AA 第三块有腿部拖影和局部透明残影，AD 第三块边界帧差较大。这个单场景/seed 结果证明了**有限的原权重减步余量**，不能据此宣布成熟画质、124 帧 8-step 稳定性、AnyFlow 成功或端到端加速。

| 路径/新增 RGB | 8-step 水平 flow px/帧 | 30-step 已存对照 | 8-step sampling / 30-step sampling | 8-step 边界灰度 MAD |
| --- | ---: | ---: | ---: | ---: |
| AA 39–55 | +0.781 | +1.347 | 70.87 / 142.91 s | 3.54 |
| AD 39–55 | −1.510 | −1.458 | 40.52 / 135.69 s | 2.81 |
| AA 56–72 | +0.978 | +0.474 | 37.76 / 155.14 s | 4.89 |
| AD 56–72 | −0.732 | −0.930 | 42.51 / 146.39 s | 7.95 |

第二块是同历史、同噪声、同条件的步数对照；第三块两侧历史已各自生成，只作闭环能力比较。30-step 与 8-step 运行时刻不同且共享硬件有其他作业，表中耗时是单次观测，不能用比值宣称严格 speedup。首屏和从零完整 73 帧的端到端时间未测。CPU KV 在第二块采样时为 **6.799 GB**，第三块采样时为 **9.632 GB**；本轮记录的最大 GPU allocated 为 **25,682 MiB**。

## 视频

左侧为已存 V3 Baseline 30-step，右侧为本轮 8-step；顶部与底部标出步骤、动作和共用/各自历史。原片与并排片均为真实 H.264、24 fps，已完整解码并检查帧数和递增 PTS。

| 路径 | 8-step 原片 | 左 30-step / 右 8-step |
| --- | --- | --- |
| AA 56 帧 | [原片](artifacts/videos/AA_8step_56.mp4) | [并排](artifacts/videos/V3_30_vs_8_AA_56.mp4) |
| AD 56 帧 | [原片](artifacts/videos/AD_8step_56.mp4) | [并排](artifacts/videos/V3_30_vs_8_AD_56.mp4) |
| AA 73 帧 | [原片](artifacts/videos/AA_8step_73.mp4) | [并排](artifacts/videos/V3_30_vs_8_AA_73.mp4) |
| AD 73 帧 | [原片](artifacts/videos/AD_8step_73.mp4) | [并排](artifacts/videos/V3_30_vs_8_AD_73.mp4) |

四组全部新增帧及抽样并排接触图在 [`artifacts/`](artifacts)；原分辨率帧也已人工查看。实际账本为 **32 sampling + 2 clean commit = 34 forward、4 VAE、0 训练更新、466.02 GPU-seconds = 0.12945 GPU-hours**，GPU 0 顺序执行，最终退出。每次采样前记录调用；四段均核对历史 cache 内存身份/storage/version 不变，commit 只追加当前 clean endpoint，RGB 前缀在 `.npy` 层逐像素不回改。

完整机器指标见 [metrics.json](metrics.json)，[原始逐块 JSON](artifacts/metrics) 与 [预算账本](artifacts/budget.json)；[来源与复现](MANIFEST.md)、[源码哈希](source_manifest.json)、[视频审计](artifacts/video_integrity.json)、[产物 SHA](artifact_manifest.json) 可追溯代码和输入。大 latent、raw KV 和未压缩 RGB 保留在外部 `H3-World/outputs/EXP-004_v3_8step/`，没有复制入 Git。

## Judge 验收

2026-10-10 accepted：8步续写AA/AD73帧有限可行性通过；画质/严格连续性PARTIAL，AA后段肢体拖影明显。首39帧借用30步结果，完整从零8步和公平E2E未测。任务已关闭，GPU停止。[最终审核](judge/FINAL_REVIEW.md) · [原Worker报告](worker_report.md) · [原Worker指标](worker_metrics_snapshot.json)。
