# EXP-005 / v1 Worker 阶段一报告：V3 Sliding Window CPU 协议

更新：2026-10-10 HKT。**执行状态：阶段一 CPU 已完成；Judge已接受已测CPU范围，GPU未批准。** 本报告只覆盖 [next_plan.md](next_plan.md) 已授权的 CPU 工作。原 EXP-004 报告逐字节保存在 [previous_state/report.md](submission/experiments/EXP-005_v3_sliding_window/previous_state/report.md)。

## 研究问题与结果

问题是在保持冻结 V3 Original H3 + released Action LoRA、Single I0、native time、own-action/current-prefix feedback 和 30-step FM 协议的前提下，能否把历史 video raw KV 限制为最近五个完整 chunk，并建立独立的 Local RoPE 候选。阶段一实现了显式 12→5 分块的 SW-G/SW-L CPU 入口和失效即拒绝的协议检查。它**没有验证 GPU 上的完整模型输出或任何视频能力**。

14 项 CPU 测试全部通过（4.24 秒）。旧 index 2–5 的 SW-G attention 输出与冻结 EXP-003 入口在相同 toy 输入、fp32 CPU backend 和 current-prefix 上下文中逐元素一致；模型调用处还逐项比较了实际传入的 current/audio/prompt/anchor、packed 位置与 action rows、两个 timestep、prefix-time flag、own-action/feedback、frame start 和 prefix 长度。真实 parking 37-latent packed 输入的时间网格、动作/video 可见性裁剪通过检查。缓存索引/容量/clean commit、SW-L 读时局部位置与 canonical global RoPE 提交、prefix 坐标保持和旧 RGB append-only 均通过小张量检查。SW-L 改变 video-to-prefix attention logits，因此不是免费或无损的缓存优化。

目前**没有经认证的 >37-latent 原生条件输入**。直接调用冻结 packed builder 扩成 42 latent 会移动旧 action/I0/video 坐标。结构性 toy 长输入只证明索引与 mask 能运行，不代表真实模型可用。`interval_sw` 对 post-37 GPU 调用保持关闭；真实 C7/C8 生成必须先解决此输入协议并单独放行。冻结 H3 路径使用 MM-RoPE；SolarWM 的 camera PRoPE 不在当前实现内。

## 执行与资源

命令：`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -m pytest -q submission/experiments/EXP-005_v3_sliding_window/test_contract.py`。原始日志为 [cpu_tests.log](submission/experiments/EXP-005_v3_sliding_window/artifacts/cpu_tests.log)，结果 `14 passed in 4.24s`。**GPU forward 0、GPU 小时 0、VAE 0、训练更新 0、生成视频 0。** 没有加载 33B 权重。

代码与证据：[README](submission/experiments/EXP-005_v3_sliding_window/README.md)、[PROTOCOL](submission/experiments/EXP-005_v3_sliding_window/PROTOCOL.md)、[MANIFEST](submission/experiments/EXP-005_v3_sliding_window/MANIFEST.md)、[metrics.json](submission/experiments/EXP-005_v3_sliding_window/metrics.json)。`GPU_PLAN.md` 和 `FUTURE_ANYFLOW.md` 是 Judge 已写的未来提案，本轮未执行、未覆盖。

## 阶段二前置条件与建议

先构造并认证原生 >37-latent packed 条件：旧 37 个视频 latent 和对应 prefix/action/I0/audio 的坐标及 embedding 必须保持，新的动作 span/位置需审计；新噪声独立生成，旧噪声逐值复制。再冻结 runner、输入和权重 hash，做 G0 同状态完整模型回归；若通过，再考虑 G1 SW-G 第 7/8 块，SW-L 另行决策。首次淘汰必须先将 EXP-003 的 C6 clean endpoint **恰好提交一次**，不能误用未提交的 C1–C5 cache。CPU 通过不代表动作、视觉或长期稳定性 PASS。

Judge归档说明：原Worker报告的资源日期推定保存在[原始快照](submission/experiments/EXP-005_v3_sliding_window/worker_report.md)。当前GPU额度为0；未来提案仅1卡、项目合计≤3，不因历史晚间8卡表述重置额度。阶段二须另批，当前未启动GPU。本报告原始执行内容由Worker提交，最终审核见[Judge结论](submission/experiments/EXP-005_v3_sliding_window/judge/FINAL_REVIEW.md)。

## Judge最终状态

阶段一CPU交付已接受；独立复跑14项通过（3.74秒）。SW-G/SW-L生成能力NOT_TESTED，真实长fixture与GPU runner待准备，阶段二未授权。
