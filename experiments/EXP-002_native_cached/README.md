# EXP-002：Native 条件下的严格缓存续写候选

**Judge accepted：73帧严格缓存候选可行性通过，正式124帧V3仍待验证。** [正式审核](judge/FINAL_REVIEW.md)。 本实验只回答一个问题：Original H3 + released action LoRA 在 Single I0、12→5→5 latent 分块下，使用 current-prefix 路由和真实 persistent raw video KV 时，能否在自身生成历史中同时保留基本人物结构和 A/D 动作响应。它是计划中 V3 的候选证据，尚不能称为正式 V3，也没有进行新训练。

AA 与 AD 共用同一个保存的 `first12_A`、首 39 RGB 帧、首窗 clean-commit KV、初始噪声、图像/audio 条件、seed 13 和全局位置。第二块只改变当前动作 A/D；第三块各自接自己的第二块生成 latent 和缓存。全程 30-step native FM，flow shift 2.22，没有 GT reset、RGB anchor、AnyFlow 或 DMD。`current-prefix` 允许公共 prefix 读取当前视频，当前 action 只反馈其对应视频；当前视频读取历史 raw video KV 和自己的 action，未来 action/video 在进入 text refiner 前物理删除。

| 路径 | RGB 39–55 当前动作 | RGB 56–72 当前动作 | 人物/场景逐帧检查 |
| --- | ---: | ---: | --- |
| AA | A；水平 flow **+1.347** px/帧 | A；**+0.474** | 两块可辨；第三块人物转身，RGB55→56 边界变化明显 |
| AD | D；**−1.458** px/帧 | D；**−0.930** | 两块可辨；有普通动作/姿态细节缺陷 |

辅助对照 V2b Same-σ 在相应新增块的 flow 是 AA `+2.146/+0.347`、AD `−1.752/−0.972`。flow 仅是运动代理；Worker 查看了全部帧接触图与边界/末帧，人物没有分解为透明多体或瞬间换场。第二块 AA/AD 的历史相同，因此动作差异具有明确的同历史反事实意义；第三块两条路径已经有各自的生成历史，不能称为同状态反事实。V2b 与本候选同时改变了历史 σ、attention 拓扑和缓存策略，因此并排视频是能力比较，**不是单因素消融**。

## 视频

| 路径 | 本候选 56/73 帧 | 左 V2b、右本候选 56/73 帧 |
| --- | --- | --- |
| AA | [56 帧](artifacts/videos/AA_rollout_56.mp4) · [73 帧](artifacts/videos/AA_rollout_73.mp4) | [56 帧](artifacts/videos/V2b_vs_cached_AA_56.mp4) · [73 帧](artifacts/videos/V2b_vs_cached_AA_73.mp4) |
| AD | [56 帧](artifacts/videos/AD_rollout_56.mp4) · [73 帧](artifacts/videos/AD_rollout_73.mp4) | [56 帧](artifacts/videos/V2b_vs_cached_AD_56.mp4) · [73 帧](artifacts/videos/V2b_vs_cached_AD_73.mp4) |

8 个 MP4 均为真实 H.264、24 fps、832×480 或并排 1664×560，逐帧解码、帧数、PTS 已检查。首次显示的 RGB 前缀在 `.npy` 层逐像素冻结：73 帧的前 56 帧与已发布 56 帧完全一致；两条 56 帧的前 39 帧一致。检查结果见 [视频审计](artifacts/video_integrity.json)。并排 MP4 经重新编码，不能用其压缩像素逐字节比较原视频。

## 实现、预算与证据

- [显式区间缓存入口](interval_cached.py) 把全局 latent `start/stop` 与 cache index 分开，使用 `fixed_prefix_timesteps=False`，沿用已有 [current-prefix 路由](../../reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py)、`H3ChunkCache` 与全局 RoPE。首 12 latent 的固定状态 velocity 与保存的 Original 参考 relative RMS、max abs 均为 **0**。
- 首窗 KV 50 层、4,680 video tokens/层，占 **6,799,104,000 bytes**；第二块各自 commit 后为 6,630 tokens/层、**9,632,064,000 bytes**。后续采样只读缓存；AD 第二块和两条第三块都记录了内存 entry 身份、storage 指针、tensor version 与 commit 计数不变。AA 第二块的旧 runner 只记录了缓存文件 SHA 不变，没有进程内身份检查；其原始源码保存为 [AA at-run 快照](run_rollout_aa_at_run.py)，没有为补这个测量重复推理。
- 完整预算是 **120 sampling + 3 prefill/commit + 1 首窗诊断 = 124 forward**，4 次 VAE decode、0 optimizer update、单卡 GPU 0 顺序运行，共 **825.24 GPU-seconds = 0.2292 GPU-hours**。最高 `torch.cuda.max_memory_allocated` **26,686 MiB**。第一条 AA 的测量在 prefill 后重置过 peak，故它的 25,498 MiB 不包含首窗 prefill；其余运行记录包含本进程内 sampling、commit、decode 的峰值。四次分进程运行的 wall 是 AA 219.2+200.8 s、AD 188.1+191.2 s；这些是本实验增量运行，不等于从零生成完整 73 帧的端到端时间，也不能据此声称比 Original 加速。
- [CPU 协议检查](test_contract.py) 使用冻结真实 parking input，9 passed；旧 current-prefix tiny-H3 10 passed。它们覆盖 action span、全局位置、未来裁剪、own-action mask、缓存追加与重放。首次运行输出只在工具记录中，未落盘；[04:20 HKT 事后 CPU 复核日志](artifacts/cpu_tests_recheck_20261010.log)保留了相同两组测试的 9+10 passed，不冒充首次运行日志。原始指标见 [metrics.json](metrics.json)、[逐块 JSON](artifacts/metrics/)、[预算](artifacts/budget.json)；大 latent、raw KV 与冻结 `.npy` 留在外部 [输出目录](/home/qma/work/GWM/H3-World/outputs/EXP-002_native_cached)，不复制到 Git。

复现代码调用顺序见 [MANIFEST.md](MANIFEST.md)。本轮停在 73 帧；没有 124 帧、多场景/seed、动作切换后的多步泛化或公平端到端速度结论。任务快照见 [taskbook_v1.md](taskbook_v1.md)，正式能力判定由 Judge 完成。
