# EXP-018/v1 Worker 报告：七版本统一 124 帧成本与视频

2026-10-13 HKT。执行范围来自根目录 `next_plan.md`，本轮只有推理和 CPU 视频校验，**0 训练更新**。七个冻结配置均从本轮新生成的噪声和现场重新编码的相同停车场 I0、prompt、持续 A 条件开始，独立生成 124 RGB 帧；各导出前 120 帧作为 24fps、5.000 秒展示片。没有读取旧首窗、旧 endpoint 或旧 KV，没有因质量筛 seed/checkpoint。所有成功行都在同一 GPU5（UUID `GPU-a2f7e751-5add-d6dd-37bd-2c91e1640edd`）串行运行，启动时报告空闲 45,453 MiB，`ABOT_VRAM_RESERVE_GIB=18`。

## 实际结果

主耗时是独立子进程启动至退出的 wall，包含加载、现场条件编码、全部生成与保存；**是完整 124 帧的成本，不是展示片的 120 帧成本**。每行一次，n=1。`NFE` 仅计采样网络求值；commit/prefill 单列。

| 行 | 协议摘要 | 子进程 wall (s) | 采样 NFE | commit | 总 DiT forward | GPU allocated 峰 (GiB) | CPU KV 峰 (GiB) | Worker 质量观察 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| V0 | Original 双向，整片 30 NFE | 802.556* | 30 | 0 | 30 | 26.258 | 0 | 原始参考；正式比较待 Judge |
| V1 | Native causal，latent dual anchor，8×8 NFE | 417.877 | 64 | 8 | 72 | 26.393 | 13.192 | 早期 causal 参考；正式比较待 Judge |
| V2a | RGB dual anchor + visual/action 适配，8×8 NFE | 658.413 | 64 | 8 | 72 | 26.453 | 13.192 | 联合视觉修复配置；正式比较待 Judge |
| V2b | Same-σ 历史重算，6×30 NFE | 2172.014 | 180 | 0 | 180 | 26.258 | 0 | 局部双向协议；正式比较待 Judge |
| V3-30 | Strict causal + raw KV，6×30 NFE | 1358.263 | 180 | 5 | 185 | 26.321 | 16.886 | 人物结构局部形变，PARTIAL |
| V3-8 | 同 V3 的普通 FM 8 NFE，6×8 | 505.067 | 48 | 5 | 53 | 26.321 | 16.886 | 腿部重影和场景漂移，PARTIAL |
| V4 | EXP-008 cycle8 DMD student，6×8 NFE | 485.456 | 48 | 5 | 53 | 26.387 | 16.886 | **全片彩噪，生成 FAIL** |

`*` V0 的计时器在生成后的视频 FPS 字符串断言处退出：PyAV 报 `24`，原断言只接受 `24/1`。模型生成和两份 MP4 均已完成；`802.556 s` 是启动时间至 124 帧原片 mtime 的恢复值，约 1 秒精度。恢复仅用已存在视频做后处理，没有重跑 GPU。其余六行 wall 由 launcher 计时器记录至子进程退出。各行 GPU reserved 峰、分阶段计时与视频 SHA 见 `results_index.json`、`metrics_table.csv`；不要把 allocated 理解为模型权重或 NVML 全卡占用。

七次成功进程合计 **6399.646 GPU 秒（1.777679 GPU 小时）**、**645 个 DiT forward**（采样与 commit 合计）。此外 V2b 第一次在条件编码、0 DiT forward 时 OOM，实际再消耗 **17.472 GPU 秒**；含失败尝试合计 **6417.118 GPU 秒（1.782532 GPU 小时）**，仍低于 4 GPU 小时上限。该失败尝试 allocated 峰 43.635 GiB；其原始 `metrics.json` 中 `peak_reserved_gib: 0.0` 是异常退出后未写成的默认值，**应视为缺测，不是测得的零**。初版 native wrapper 漏掉冻结协议要求的 `torch.no_grad()`；修复时也调整了 FP32 DiT 准备与文本编码的先后顺序，未单独验证两项改动各自对显存的贡献。修正入口后仅重试一次，未改变权重、噪声、精度档或 solver。失败实物完整保留在 `H3-World/outputs/EXP-018_unified_5s_benchmark/v2b_attempt_01_condition_oom/`，仓库副本位于 `artifacts/v2b_attempt_01_condition_oom/`。

## 输入、实现与视频核查

- 所有行的 full37 video noise 原始字节 SHA-256 相同：`805e1c307512549fdaafe6e208bdee92819d16288b217f14201233a59a385794`。每行独立进程生成；native 行的旧 fixture **仅作逐值协议核对**，未被读取为生成端点或历史 KV。原始 action、I0、noise、adapter 与实现 SHA 见 `SOURCE_MANIFEST.json` 和 `RUNTIME_DEPENDENCIES.json`，各行实际命令/环境在 `results_index.json`、对应 `launch.json`。
- 每行 `full_video` 实际解码 **124 帧**，`display_video` 实际解码 **120 帧**；14 份 MP4 均为 832×480、24fps，PTS 有序且唯一，完整结果与文件 SHA 在 `media_validation.json`。复核脚本为 `validate_media.py`。原片保留在 `H3-World/outputs/EXP-018_unified_5s_benchmark/<row>/`，并逐字节复制到本目录 `artifacts/<row>/`；48 份源/副本 SHA 见 `ARTIFACT_MANIFEST.json`，没有覆盖旧视频。120 帧展示副本位于 `submission/report/unified_5s/videos/`。
- V4 片首、中、尾抽样均为全画面彩色噪声，人物和停车场不可辨；这条行只证明现有权重/入口完成了 124 帧数值执行，**不能称为 Stage2 质量成功**。V3-30 在约 80–91 帧有瞬态身体折叠、108 帧有残影；V3-8 腿部持续重影并伴场景漂移。画面质量以原片及 Judge 后续正式验收为准，单条持续 A 视频不能验证 A/D 动作分离。

## 公平性与解释边界

相同的是初始图像、prompt、持续 A、seed13、832×480、24fps、flow shift 2.22、完整 37-latent 噪声及 124 帧长度。保留各版本自己的 anchor、attention、历史协议、adapter、chunk partition、precision 和采样次数，因此这是**完整方案成本与能力对照，不是单因素消融**。V0 是整片一次 30 NFE；V1/V2a 是 8 块各 8 NFE；V2b/V3-30 是 6 块各 30 NFE；V3-8/V4 是 6 块各 8 NFE。V1/V2a 按冻结实现有 8 次 commit；V3/V4 最后一块无需后继读取，仅有 5 次 commit。V3-8 是普通 flow matching 减步，不是 AnyFlow。

V0 首次读取模型文件，加载与条件准备分别耗时约 63.7 和 138.8 秒；后续行因系统文件缓存明显缩短。V2a 七次在线 RGB anchor 处理额外约 170.7 秒。V2b 的历史重算使逐块采样时间增长。这些差异与冷启动 n=1 的缓存/磁盘/CPU波动共同存在，**不能仅据 wall 比值归因为 causal/KV 的纯算法加速**。`ready_to_video_seconds` 与早期 legacy 的 `sampling_loop_inclusive_seconds` 阶段边界不完全相同，因此主表只采用相同的子进程 wall 口径。

本轮预算和交付已完成；不追加训练、推理或 seed 扫描。需要由 Judge 对七条实物视频进行最终质量/动作解释、制作并排总览和 Git 发布。完整原始数值以 `results_index.json`、`metrics_table.csv`、各行 `native_metrics.json` 或 legacy metrics 为准；Worker 报告不改写这些测量。
