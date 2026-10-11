# EXP-015/v1 · 四段真实 56 帧数据与 native 动作

状态：**CPU 数据与动作完整性已通过；模型编码和训练尚未执行。** 本目录以 EXP-013 冻结的四个 train episode 为输入，生成四段真实录屏的 56 RGB 帧（24 FPS、832×480）及各自的新 I0。每段完整 56 帧先映射到 H3 原生 17 个 temporal latents，再按 **12+5** 分为 C1/C2。没有 GT reset、反事实 D 真值或新模型结果。

| 场景 | 真实按键（56帧） | C2 实际动作 | 视频 |
| --- | --- | --- | --- |
| 43866101 | A+S+L | 后退+左移、相机慢速向右 | [MP4](artifacts/43866101/real56.mp4) |
| 7199292c | A+S | 后退+左移 | [MP4](artifacts/7199292c/real56.mp4) |
| 9dc2e588 | A+S+J | 后退+左移、相机向左 | [MP4](artifacts/9dc2e588/real56.mp4) |
| b784d995 | A 40帧、静止8帧、W 8帧 | A→静止→W | [MP4](artifacts/b784d995/real56.mp4) |

证据入口：[Worker 报告](WORKER_REPORT.md) · [完整输出 manifest](OUTPUT_MANIFEST.json) · [实际源 PTS 审计](SOURCE_PTS_AUDIT.json) · [近邻像素抽查及局限](SOURCE_ALIGNMENT_PIXEL_SPOTCHECK.json) · [Judge 独立动作核查](judge/INDEPENDENT_DATA_AUDIT.json)。每个场景的 `I0.png`、`raw56.npy`、`pooled17.npy`、`keys9_17.npy` 和 `action_script17.json` 保存在同名 `artifacts/` 子目录；路径、SHA、shape、来源与 FFmpeg 命令见 manifest。

本实验只准备数据。F 是从观测 J/L 与 COLMAP yaw 速度派生的 fast-pan，不等于录屏 Space；Q/E/Space 在这四段均未按下。真实混合键保持原样，未改写成纯 A/D。后续 VAE C1 前缀一致性必须在同一新 56 帧视频上单独验证；不能用旧 39 帧 MP4 的重编码差异推断未来泄漏。

[Judge最终验收](judge/FINAL_REVIEW.md)：数据准备ACCEPT WITH LIMITS；下一项EXP-016仅VAE验证。原始numpy数组留本机manifest路径，不纳入Git，可由固定源和build_real56.py重建。变长动作文本的packed位置未来隔离尚待独立解决，不能把数据准备写成完整训练fixture通过。
