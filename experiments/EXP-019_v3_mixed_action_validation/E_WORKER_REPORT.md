# EXP-019/v1 · E 阶段 Worker 报告

2026-10-15 HKT。Judge 已在 `judge/R_REVIEW_AND_E_RELEASE.md` 接受 R 并签发 `judge/E_APPROVED.json`。本 Worker 在 GPU0 空闲、E 输出目录不存在、marker 与 `SOURCE_MANIFEST.json` SHA 一致后，运行冻结 `run.py --stage E`。没有改动 R/E 的模型入口，没有启动 G1/G2、训练或 DiT 推理。

## 实际执行与资源

- 命令：`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-019_v3_mixed_action_validation/run.py --stage E --gpu 0`，stdout/stderr 在 `artifacts/E/E.log`。
- 同一 NVIDIA L40 GPU0，UUID `GPU-d77c5635-02ec-c2d2-d744-1dfcc4a14e8f`，启动空闲 45,458 MiB；`ABOT_VRAM_RESERVE_GIB=18`。冻结源码清单 SHA-256 `5837a43755deea41db9f0c1cdefb0a0e31afa9fb7bbbb6e44a414b153570fb3e`。
- 账本 `complete`，**14 次 text encoder 调用，0 DiT forward、0 VAE encode/decode、0 训练更新**。完整 wall **192.620 秒**（阶段上限 900 秒）；PyTorch allocated 峰 **26.483 GiB**，reserved 峰 **27.004 GiB**。独立 900 秒 watchdog 在进程正常退出后结束，没有发送信号。
- 四个 `.pt` fixture 从本轮真实 I0 PNG、原 scene caption 及 17 句真实混合动作脚本现场生成；CPU 审计确认它们的 I0 SHA 与 EXP-015 真实录屏来源一致，并且**不同于** EXP-014 canonical 模板中的旧首帧。新 Qwen 图文 head 也与旧 fixture head 数值不同，未拿旧条件冒充新条件。初始 video/audio noise 沿用冻结模板并逐值核对。

| Scene | 新 head 行数 | 前 17 个真实动作 span 长度 | Fixture SHA-256 前缀 | Anchor SHA-256 前缀 |
|---|---:|---|---|---|
| `43866101` | 548 | 14 × 17 | `d343a6a98e0b` | `c01e122ff32c` |
| `7199292c` | 565 | 13 × 17 | `860f8219f16a` | `661ffd4eb753` |
| `9dc2e588` | 572 | 14 × 17 | `5359339c412b` | `e0480c16fed4` |
| `b784d995` | 567 | 10 × 13，8 × 4 | `0347bbdd9c78` | `3c6f973aab18` |

完整 SHA、输出路径与逐项检查在 `artifacts/E/result.json`、`E_CPU_AUDIT.json` 和 `E_ARTIFACT_MANIFEST.json`。四个 packed 布局均有 37 个连续 action spans，与 `action_text_rows` 一致；真实前 17 span 的长度与上表相符。文本 `token_tags` 中动作行全部为 1，prompt 与布局文本行数一致且数值有限。

## Future isolation 与原生图像条件

CPU 独立核查在 `stop=12` 和 `stop=17` 各作一次未来动作干预：保留同一真实已编码 head/过去动作 embedding，未来改用**当前 fixture 中另一条实际独立 Qwen 编码的动作句 embedding**，并重新构造相应 token 长度的 packed 布局。四个 scene 的 8 次干预均改变未来内容和 token 长度；当前可见的 prompt 行与布局逐值相同，历史语义坐标稳定。这验证了保存的真实 embedding 在裁剪后不受此类未来动作内容/长度影响；没有为反事实未来再调用一次 Qwen，也没有验证模型生成后果。

原生 keyframe conditioner 配置字段为 `imgvid_cond_noise_aug=0.999`。冻结 pipeline 用 `torch.tensor(0.999, dtype=torch.bfloat16)` 计算混合系数，该值在 bfloat16 **实际等于 1.0**，噪声权重等于 0。四个 E anchor 都按冻结公式从本轮 I0 latent、seed13 噪声独立重算并逐值相同，同时逐值等于 clean I0 patch rows。此数值事实也适用于使用同一冻结公式的 EXP-014；本轮没有修改精度或重新定义协议。后续报告应称“配置为 .999、实际为 clean I0”，不能声称数值上注入了 0.1% 噪声。

## G 阶段 CPU 环境准备

冻结 `run.py` 的视频读写通过系统命令 `ffmpeg`，但当前系统 PATH 无该命令。按 Judge 指示，只在任务输出目录建立 `bin/ffmpeg` 链接，指向现有 `imageio_ffmpeg` 7.0.2 二进制，**没有修改冻结 runner、模型或采样协议**。CPU 检查先完整解码已有 56 帧真实源视频，并编码/回读 8 帧短片；再直接调用冻结 `run.py` 的 `read_video`/`save_video` 函数，完成 56 帧读取与 8 帧编码回读，两项均通过。链接 realpath、二进制 SHA、版本、启动 PATH 前缀与媒体 SHA 见 `FFMPEG_CPU_PREFLIGHT.json`、`G_CPU_IO_PREFLIGHT.json`；准备脚本为 `prepare_ffmpeg.py`。未来 G 命令需要把该 `bin` 目录放在 PATH 首位；本段检查不构成 G 的 GPU 放行。

## 结论与边界

E 验证了四场景真实首帧/文本条件与变长动作 span 的编码、当前可见 embedding 的未来隔离及 frozen anchor 计算。它不证明混合动作已经影响视频 velocity 或实际续写质量。四段来自训练 episodes，且没有完整 D 反转 GT。下一步 G1 需等 Judge 审核 E 并签发 marker；G1 只生成真实历史之后的 17 帧，不能把前 39 帧 GT 展示说成模型从零生成。
