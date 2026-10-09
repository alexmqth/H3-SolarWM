# EXP-002 / v1 Worker 执行报告：Native 严格缓存候选 73 帧

更新：2026-10-10 04:20 Asia/Hong_Kong。**Worker Status: completed；Judge Acceptance: pending。** 本报告仅提交执行事实与 Worker 判断，不宣布正式 V3 已完成。任务依据 [next_plan.md](next_plan.md)，详细材料位于 [EXP-002 实验目录](submission/experiments/EXP-002_native_cached/README.md)。

## 研究问题与结果

本轮验证 Original H3 + released action LoRA 在 Single I0、native 时间、12→5→5 latent 分块下，用 current-prefix 严格分块因果路由和真实 persistent raw video KV，是否能在自身生成历史里保留人物结构和 A/D 动作响应。AA 与 AD 共享同一个已保存 `first12_A` 生成首窗、同一首窗 clean-commit KV、首 39 RGB、初始 noise、anchor、audio、seed 13、全局 RoPE 和 30-step/块；第二块唯一动作差别是 A 对 D。第三块接各自的第二块生成 latent 与 committed KV，没有 GT 或 V2b 后续块重置。本轮没有训练、AnyFlow、DMD 或新基线推理。

**有限正结果：** 两条路径均实际生成并逐帧解码至 **73 RGB 帧**。第二块（RGB39–55）AA/AD 水平 Farneback flow 为 **+1.347/−1.458 px/帧**；第三块（RGB56–72）为 **+0.474/−0.930**。方向在两块持续区分，人物保持单体可辨、停车场结构延续，没有早期路线的透明分解或瞬间换场。AA 第三块人物转身，边界灰度 MAD **15.01**，明显高于本块帧内 **3.83**；视觉质量不能写成无瑕或长期稳定。AD 第三块边界 MAD **6.32**。flow 是辅助运动代理，不是严格动作正确率。第二块是同历史反事实；第三块的两条历史已分化，只证明各自持续动作在自身历史中仍有响应。

| 新增块 | AA flow | AD flow | AA/AD 边界灰度 MAD | 两条片段状态 |
| --- | ---: | ---: | ---: | --- |
| RGB39–55 | +1.347 | −1.458 | 3.48 / 3.57 | 人物和场景可辨，动作方向分化 |
| RGB56–72 | +0.474 | −0.930 | 15.01 / 6.32 | 人物仍可辨，AA 有明显边界/姿态变化 |

参考 V2b Same-σ 局部双向重算在相应块的 flow 为 AA +2.146/+0.347、AD −1.752/−0.972；本候选不是从 V2b checkpoint 顺承，也不是单变量替换，故这些数值只能说明能力范围，不能推断严格缓存单独造成改善或退化。Original A→D 匹配片仍缺失，本轮按任务书没有新增。

## 协议与缓存证据

新 [interval_cached.py](submission/experiments/EXP-002_native_cached/interval_cached.py) 将全局 latent start/stop 与 cache index 分离，用原生 `fixed_prefix_timesteps=False`，输入进入 refiner 前由 `visible_inputs` 物理删除未来 action/video；current-prefix 路由来自已有候选，没有改 H3 权重。CPU 冻结真实输入检查 **9 passed**，旧 current-prefix tiny-H3 测试 **10 passed**。首次 CPU 测试输出只在工具记录中，未落盘；[04:20 HKT 的事后复核日志](submission/experiments/EXP-002_native_cached/artifacts/cpu_tests_recheck_20261010.log)记录了同两组测试再次 **9+10 passed**，不冒充首次日志。真实 H3 首12固定 noisy state 的候选 velocity 与保存 Original 参考 relative RMS/max abs 都是 **0**，说明首窗身份没有引入数值差异。

首窗 raw KV 为 50 层、每层 4,680 video tokens，**6,799,104,000 bytes**；第二块各自 clean-commit 后每层累计 6,630 tokens，**9,632,064,000 bytes**。共同首窗 KV 文件 SHA 两路径一致，第二块后各自缓存 SHA 不同。AD 第二块和两条第三块的采样前后，内存 entry 身份、storage 指针、tensor version、commit 数都保持不变；历史没有在每个 sigma 重算。已发布 RGB 前缀也经 `.npy` 逐像素核对：两路径首39完全一致，每条73片的前56与原已发布56帧完全一致。AA 第二块在增强 instrumentation 前执行，仅记录磁盘首窗 KV 文件 hash 不变，没有进程内 identity 检查；原运行源码已冻结为 [AA at-run 快照](submission/experiments/EXP-002_native_cached/run_rollout_aa_at_run.py)，没有为补这一个测量重跑 GPU。

## 视频、完整性与预算

- [AA 73 帧原片](submission/experiments/EXP-002_native_cached/artifacts/videos/AA_rollout_73.mp4)、[AD 73 帧原片](submission/experiments/EXP-002_native_cached/artifacts/videos/AD_rollout_73.mp4)。56 帧原片也在实验 README 中。
- [左 V2b、右 strict KV：AA 73 帧](submission/experiments/EXP-002_native_cached/artifacts/videos/V2b_vs_cached_AA_73.mp4)、[AD 73 帧](submission/experiments/EXP-002_native_cached/artifacts/videos/V2b_vs_cached_AD_73.mp4)；56 帧并排片同样已交付。两侧帧数、24fps、可视时间一致，顶部标明协议，底部标动作与块边界。对比是跨协议能力比较：V2b 同σ历史联合重算，候选为 sigma0 hidden KV 与严格分块因果。新增块之后两侧历史不再相同。
- 8 个 MP4 逐帧解码、H.264、24 fps、尺寸、PTS 递增与帧数已通过 [完整性审计](submission/experiments/EXP-002_native_cached/artifacts/video_integrity.json)。原视频没有被覆盖；小 MP4、JSON 已复制到 submission，大 tensor/KV 保留在外部输出目录并由 [artifact manifest](submission/experiments/EXP-002_native_cached/artifact_manifest.json) 关联。

预算实际消耗：**120 sampling + 3 prefill/commit + 1 identity diagnostic = 124 次 denoiser forward**，4 次 VAE decode，0 optimizer update；GPU 0 顺序执行四个进程，合计 **825.24 GPU-seconds / 0.2292 GPU-hours**，首至末约 15 分钟，峰值 `torch.cuda.max_memory_allocated` **26,686 MiB**，均未触及任务上限。第一条 AA 运行在 prefill 后重置了 peak，其单条 25,498 MiB 未包含 prefill 峰值；后续三条完整记录其各自 sampling、commit、decode 的进程内峰值。四次运行单块 wall 分别为 AA 219.2/200.8 s、AD 188.1/191.2 s，包含每次模型加载和当前块处理，但不等于从零 73 帧端到端时间，不能与 Original 30-step 全片时间直接相除得加速比。逐块原始指标和预算见 [metrics.json](submission/experiments/EXP-002_native_cached/metrics.json) 与 [budget.json](submission/experiments/EXP-002_native_cached/artifacts/budget.json)。

## Worker 判断与边界

这是一条**有信息价值的 V3 可行性候选**：首窗 identity、严格未来隔离、真实 raw KV 复用、同历史 A/D 第二块分化以及两条 73 帧基本视觉结构同时得到证据。它仍不是正式高效 V3 的完整证明：只测了单场景/seed、持续 A 或先 A 后持续 D 到 73 帧；没有 124 帧、其它动作、多场景或公平端到端加速验证。第三块 AA 的边界跳变也需保留为限制。按任务书本轮停在 73 帧，后续扩展或正式版本判定交 Judge 决定；Worker 不自动启动新实验。
