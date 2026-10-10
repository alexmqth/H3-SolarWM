# EXP-008/v2 DMD8 视频评估 — Worker 负结果报告

2026-10-11 HKT。状态：**cycle8 AA C2视觉失败，按停止条件终止评估；等待Judge最终审核**。本轮没有C3视频，没有AD完整视频，没有重试、重新训练、换checkpoint或调参。

## 固定比较协议

使用v2训练最终且唯一预定的`cycle_08/` student QKV + target-time，native 8 finite-map steps、shift2.22、V3 strict causal/Global RoPE/persistent CPU raw KV、Single I0/current-prefix。与EXP-006 FM8和EXP-007 AF8共享同一新生成的首12 latent/前39 RGB帧、seed13、动作和C2初始噪声；DMD8用自身权重prefill C1 KV。C2因此可比较同一clean history下的续写；FM8/AF8/DMD8权重和训练不同，不能作单变量DMD消融。冻结来源见[dmd_eval/source_manifest_eval.json](dmd_eval/source_manifest_eval.json)，GPU阶段由Judge另行放行。

按计划顺序先完成C1 prefill，再运行AA C2。AA C2共8 sampling +1 clean commit+1 VAE，输出56帧H.264视频。人工查看全部新增17帧：**第39帧开始连续全画面彩色噪声，人物与停车场完全不可辨**。原视频可完整解码，前39帧raw RGB与FM8逐值相同；失败不是MP4容器错误、输出拼接或前缀被覆盖。AA指标：边界灰度MAD 37.157、块内27.843；AF8相应4.316/3.740，普通FM8相应3.658/4.147。彩色噪声条件下的水平光流−0.025px不具动作含义，不能用它判断A控制。

| 证据 | 文件 |
| --- | --- |
| FM8 / AF8 / DMD8 三列56帧AA对比 | [H.264视频](dmd_eval/artifacts/failure_comparison/AA_FM8_AF8_DMD8_cycle8_collapse_56.mp4) |
| 四个关键时间点 | [关键帧图](dmd_eval/artifacts/failure_comparison/AA_FM8_AF8_DMD8_cycle8_collapse_56.jpg) |
| DMD8 AA原片 | [56帧视频](dmd_eval/artifacts/raw/AA/rollout_56.mp4) |
| 全部新增17帧 | [contact sheet](dmd_eval/artifacts/raw/AA/chunk_12_17_all_frames.jpg) |
| 逐调用、资源、视频指标 | [budget](dmd_eval/artifacts/raw/budget.json) · [AA JSON](dmd_eval/artifacts/raw/AA/second_12_17.json) |
| CPU latent端点诊断 | [JSON](dmd_eval/failure_latent_diagnostic.json) · [脚本](dmd_eval/analyze_failure.py) |

## 停止与实际资源

收到“持续主体/场景崩坏则停止C3”的条件后，Worker在AA第二块完成时判断失败。AD第二块已在独立停止指令到达前启动；指令交叉期间Worker向PID发送SIGINT，随后Judge说明已启动AD本可完成，但进程已经退出（exit130）。AD仅完成5/8个sampling step，第6个forward已预记账并在中途被中断，`AD/second_12_17.json`状态为`failed`、错误`KeyboardInterrupt()`；**没有AD视频、不能称AD效果已测**。没有自动重启AD。AA/AD第三块均未启动。原始AD日志、失败JSON和预算全部保留。

累计实际预算为**16 forward = 14 sampling（AA8 + AD第6次已计费）+2 clean commit、1 VAE、0 update，GPU0 205.104秒（0.05697 GPU小时）**，低于批准35forward/4VAE/0.35GPU小时。AA第二块allocated峰值25.970GiB；C1 CPU raw KV 6.33GiB、AA C2 commit后约8.97GiB。AD没有完成本段显存测量。原始输出位于`H3-World/outputs/EXP-008_v3_dmd_eval/`，提交包只复制小日志、JSON和可播放视频；[证据manifest](dmd_eval/failure_evidence_manifest.json)保留SHA与外部checkpoint关联。

## 失败机制的有限诊断

在相同C2初始噪声上，DMD8保存的latent终点与初噪声cosine为**0.949**，FM8为0.100、AF8为0.114。DMD8水平latent总变差1.141，接近初噪声1.125；FM8/AF8只有0.224/0.236。DMD8端点与AF8端点cosine为0.038。这直接支持**DMD8采样没有有效去噪，噪声式latent进入VAE后形成全画面彩噪**。终点的全局均值/标准差仍与基线接近，所以仅看latent RMS会漏掉这一失败。

v2训练中fake loss在cycle4跃升到31.53，后几轮student梯度明显变小；二者和视频失败同时存在，但还不能仅凭这些数据确定唯一根因。可能涉及fake score跟随、DMD方向归一化、有限数据/更新策略或target-time地图被推离可用区间。没有做单因素实验，也没有从中间checkpoint挑片。**真实33B DMD cycle训练可执行，但当前cycle8模型能力失败；不能将其作为Stage2成功演示。**
