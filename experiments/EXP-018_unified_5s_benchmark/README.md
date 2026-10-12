# EXP-018/v1 · V0–V4 统一 124 帧生成与约 5 秒展示

任务书：[taskbook_v1.md](taskbook_v1.md)。本目录是原始执行证据，不代表 Judge 已验收画质。七行各从 seed13 的完整 37-latent 噪声独立生成 124 RGB 帧；前 120 帧导出为 24fps、严格 5.000 秒的展示片。每行只运行一次，不挑选 seed、checkpoint 或输出。

| ID | 冻结协议 | 实际采样口径 |
|---|---|---|
| `v0` | Original 双向，released action LoRA，Single I0 | 全片 30 NFE |
| `v1` | 原生早期 5-latent causal，latent dual anchor，persistent CPU KV，无新增 adapter | 8 chunks × 8 NFE |
| `v2a` | RGB dual anchor + visual QKV/action residual 联合方案，persistent CPU KV | 8 chunks × 8 NFE |
| `v2b` | Same-σ history，12+5×5，局部双向历史重算，无 persistent KV | 6 chunks × 30 NFE |
| `v3_30` | Strict causal，Global RoPE，Single I0，persistent CPU raw KV | 6 chunks × 30 NFE |
| `v3_8` | 上一行的普通 FM 8 NFE，不是 AnyFlow | 6 chunks × 8 NFE |
| `v4` | EXP-008 cycle8 DMD student 的原 checkpoint，target-time finite map，V3 causal 协议 | 6 chunks × 8 NFE |

V0/V1/V2a 使用冻结 `H3-World/code/causal/benchmark.py`；V2b/V3/V4 使用本目录 `native_row.py`。**七行都在各自独立进程内重新执行 H3 文本/I0 条件编码并生成 seed13 噪声**。native 行仅把 EXP-001 的 full37 单 I0 fixture 当作协议见证，逐值核查新编码的噪声、anchor、layout 和 prompt，不读取历史生成 endpoint 或 KV。这样冷启动 `cold_total_seconds` 都包含条件准备；各版本的 attention、anchor、precision 和 chunk 仍是不同协议，不能据此声称单变量加速。所有行显式设置 `ABOT_VRAM_RESERVE_GIB=18`，CPU raw KV 单列，旧 legacy 与 h3_fp32 精度差异保留。

CPU 预检后、任何 native GPU forward 前，为补齐统一冷启动口径修订了 native runner；初始清单保留为 `SOURCE_MANIFEST_initial_preflight.json`，正式清单的 `pre_forward_amendment` 说明差异。已启动 V0 的启动账本仍指向初始清单哈希。

**七行执行已结束，Judge 已接受本轮交付，质量限制见 [最终审核](judge/FINAL_REVIEW.md)。** 输入/权重和代码哈希见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)，逐行结果索引见 [results_index.json](results_index.json)，统一表格见 [metrics_table.csv](metrics_table.csv)，执行过程及限制见 [WORKER_REPORT.md](WORKER_REPORT.md)。完整运行输出在 `H3-World/outputs/EXP-018_unified_5s_benchmark/<ID>/`，每行包含 `launch.json`、`run.log`、原始 `*.json`、统一 `metrics.json`、124 帧原片及 `display_120.mp4`。V0/V1/V2a 的原片沿用冻结引擎命名 `baseline.mp4`/`cached.mp4`；其余为 `full_124.mp4`。所有 14 份 MP4 的完整解码、FPS、分辨率、PTS 与 SHA 核查见 [media_validation.json](media_validation.json)。

为让仓库本身保留原始证据，七条**124 帧原片**、逐行启动记录/日志/指标，以及 V2b 的首次 OOM 失败记录已逐字节复制到 [artifacts/](artifacts/)；48 份文件的源路径、目标路径、大小与 SHA-256 在 [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)。复制脚本 [archive_artifacts.py](archive_artifacts.py) 遇到同名异内容会拒绝覆盖。120 帧展示视频已在 [汇报目录](../../report/unified_5s/README.md) 单独保存，实验目录不再重复复制。

原片入口：[V0](artifacts/v0/baseline.mp4) · [V1](artifacts/v1/cached.mp4) · [V2a](artifacts/v2a/cached.mp4) · [V2b](artifacts/v2b/full_124.mp4) · [V3-30](artifacts/v3_30/full_124.mp4) · [V3-8](artifacts/v3_8/full_124.mp4) · [V4 彩噪负结果](artifacts/v4/full_124.mp4)。

V4 完整执行到 124 帧，但从首帧起是全画面彩噪，生成质量 **FAIL**；不可写成 DMD 成功。V2b 首次条件编码 OOM 的原始失败尝试也已保留，成功行由一次明确原因修正后的重试产生。七个成功进程总计 6399.646 GPU 秒、645 个 DiT forward；加上 V2b 失败尝试共 6417.118 GPU 秒、仍为 645 个 DiT forward。保留这些真实失败与计费记录，不把 smoke/执行成功写成质量验收。
