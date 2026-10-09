# EXP-003 / v1 Worker 执行报告：严格缓存候选续写至124帧

更新：2026-10-10 04:58 HKT。**Worker Status: completed；Judge Acceptance: pending。** 本报告提交本轮实际执行与证据；正式 V3 可行性版本是否验收由 Judge 决定。任务依据 [next_plan.md](next_plan.md)，完整目录见 [EXP-003 README](submission/experiments/EXP-003_native_cached_124/README.md)。本轮结束后 GPU 进程均已退出。

## 结论与协议

在 EXP-002 已验收的 Original H3 + released action LoRA、native Single I0、current-prefix own-action routing、严格分块因果、CPU persistent raw KV、30 native FM steps/chunk、seed13、停车场单场景的**同一配置**下，AA 和 AD 均从自己已生成的73帧 endpoint/cache 续写至 **124 RGB帧 / 24fps / 5.17秒**。先用各自保存的第三块 clean endpoint 在原始条件下提交一次 KV，然后各生成三个5-latent chunk；每块采样只读祖先KV，进入下一块时只提交当前 clean endpoint，末块不做无用commit。未来action/video在进入refiner前物理删除，保留全局RoPE/原生时间和同一I0，没有teacher/GT reset、RGB anchor、训练、AnyFlow或DMD。

两个124帧原片中人物与停车场到末尾仍可辨，A与D在新增三个chunk的水平光流代理符号持续相反。**但 AA 在约RGB79–86出现明显人体形变和游离肢体残影**，约RGB108–110另有短暂背景残影；AD约RGB91有短暂肢体残影。AA第五、第六块的人物恢复为单体，缺陷没有持续累积。故这支持“单场景/seed下有明确质量限制的长窗口严格缓存可行性”，不支持全帧高质量、10/20秒稳定、多场景泛化或成熟交互体验的声明。flow只是运动代理，不等于动作语义判分。EXP-002第二块的AA/AD同历史反事实仍是动作控制的更强证据；本轮73帧后两路径历史已经不同，不能再称同状态反事实。

| 新增RGB范围 | AA水平flow px/帧 | AD水平flow px/帧 | AA/AD边界灰度MAD | 实际观察 |
| --- | ---: | ---: | ---: | --- |
| 73–89 | +1.463 | −1.304 | 13.82 / 7.18 | AA约79–86明显人体形变/残影，AD基本完整 |
| 90–106 | +0.953 | −0.648 | 9.69 / 3.34 | AA恢复单体；AD约91帧短暂残影 |
| 107–123 | +1.335 | −1.240 | 2.91 / 2.50 | 两条基本结构可用；AA短暂背景残影 |

## 原始视频与对照

- [候选 AA 124帧](submission/experiments/EXP-003_native_cached_124/artifacts/videos/AA_rollout_124.mp4)和[候选 AD 124帧](submission/experiments/EXP-003_native_cached_124/artifacts/videos/AD_rollout_124.mp4)；90/107帧中间片也在 [实验 README](submission/experiments/EXP-003_native_cached_124/README.md) 中。
- [左 Original H3 持续A、右候选 AA 124](submission/experiments/EXP-003_native_cached_124/artifacts/videos/Original_vs_candidate_AA_124.mp4)；[左 V2b AA、右候选 AA 124](submission/experiments/EXP-003_native_cached_124/artifacts/videos/V2b_vs_candidate_AA_124.mp4)。两片来自真实保存的MP4，显示动作、协议与不同计时口径。
- [左 V2b AD、右候选 AD 共73帧](submission/experiments/EXP-003_native_cached_124/artifacts/videos/V2b_vs_candidate_AD_73.mp4)复用EXP-002已验收对照。Original A→D匹配片缺失，未用Original持续D冒充，也未增加基线推理。

Original整片30-step与候选每块30-step不是相同denoiser调用数；对照是能力比较而非单因素消融。Original A整片E2E为478.4s；候选AA的73→124帧进程增量wall为798.9s，含加载、cache IO、采样、提交、VAE、编码及人工检查等待，**不含旧73帧生成**，不能相除得端到端speedup。V2b AA已记录三个续写块sampling为310.26/377.53/447.19s；候选为148.45/176.43/200.42s。该观察包含拓扑、历史协议、KV策略和计时instrumentation差异，不能单独归功于KV。细节见 [效率表](submission/experiments/EXP-003_native_cached_124/efficiency_table.md)及[原始指标](submission/experiments/EXP-003_native_cached_124/metrics.json)。

## 实现正确性、预算及限制

[独立入口](submission/experiments/EXP-003_native_cached_124/interval_cached.py)显式使用全局区间 `[17,22)→[22,27)→[27,32)→[32,37)` 和独立cache index；前者只用于已保存third5提交，后面三段用于新采样。[计时路由](submission/experiments/EXP-003_native_cached_124/metered_prefix.py)只测已有current-prefix路由的历史KV读取/搬运，CPU测试与原路由逐元素输出相同。冻结的18份来源文件hash和两条EXP-002缓存的50层/token覆盖经 [CPU预检](submission/experiments/EXP-003_native_cached_124/preflight_cpu.log)核对；新增 [11项CPU测试](submission/experiments/EXP-003_native_cached_124/artifacts/cpu_tests_postrun.log)通过。

六段采样前后历史cache的entry身份、storage地址、tensor version与提交计数未变；每次commit后只追加当前chunk，最后块无commit。首次已发布73帧、90/107帧前缀在`.npy`层逐像素保持，endpoint/cache文件SHA与 [15份MP4及片段完整性审计](submission/experiments/EXP-003_native_cached_124/artifacts/video_integrity.json)均通过：H.264、24fps、帧数、尺寸、PTS递增。大latent、原始raw KV与未压缩RGB保留在外部输出目录，提交包用 [MANIFEST](submission/experiments/EXP-003_native_cached_124/MANIFEST.md) 与 [artifact manifest](submission/experiments/EXP-003_native_cached_124/artifact_manifest.json)关联，未复制33B权重或数十GB缓存入Git。

任务账本实际消耗 **180 sampling + 6 clean commit = 186完整denoiser forwards**，6次VAE，0 optimizer update，GPU 0顺序执行AA/AD，总 **1572.27 GPU-seconds / 0.43674 GPU-hours**，低于2 GPU-hours与3小时elapsed上限。记录到的新增采样块/commit/VAE最大`torch.cuda.max_memory_allocated`为 **26,876.70 MiB**；开头third5 clean commit只执行44GiB上限断言、未保存精确峰值，因此不能声称这是完整进程峰值。CPU KV从第三块提交后的12.465GB增长到末块采样前18.131GB。KV GPU transfer时间是sampling的组成部分，不可重复相加。没有从零完整124帧E2E、首块延迟或多次warmup均值。

Worker判断：任务协议有效，双路径124帧有动作与基本结构的有限正证据，且效率记录证实真实历史KV复用；AA中段明显结构缺陷必须随结果公开。是否将其验收为正式V3**可行性版本**，以及下一步是质量适配还是其它路线，交Judge根据这组完整证据决定。本轮不再启动GPU诊断或修复。
