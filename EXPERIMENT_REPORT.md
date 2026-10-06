# 实验报告

## 2026-10-07：动作/视觉取舍与 10/20 秒对照

已保留 fixed-mix 旧 action adapter 和当前 RGB visual adapter，三列原始/旧版/稳定版视频见 [动作/视觉对照](meeting/action_vs_stability/README.md)。同一 RGB checkpoint 完成 243/481 帧 W rollout，并与匹配输入的 30-step 原始 H3 对比，详见 [长视频报告和指标](meeting/long_horizon/README.md)。**20 秒 causal 视觉稳定性明确失败**：约 10 秒严重雾化，15 秒后人物和场景难以辨认，末尾退化成模糊色块。10 秒样本后段也有 blur/ghosting。两组都保留完整后段；“稳定版”仅表示 124 帧相对旧版的改善，也不证明 A/D 控制恢复。单次并发测量不作严格效率排名。

## 0. 会议主视频的纠错说明（2026-10-07）

第一次整理会议包时把旧 fixed-mix action grid 当作主视频。那套运行没有 visual tail16 adapter，并使用 latent-only dual anchor；它可以解码，但 generated-history 后段会出现人物分解和车库 tearing。当前主视频改为 `meeting/annotated/h3world_rgb_stable_action_grid_124_timed.mp4`，使用 RGB-consistent dual anchor + tail16 visual QKV adapter。视觉结构明显更稳定，但 A/D flow 仍未通过 gate；这次更换同时改变了多个协议，因此只能报告为组合协议的视觉修复，不能把改善归因于单个 anchor 或 adapter。

当前 RGB-main 的 124-frame A/D flow 为 A=`-0.784`、D=`-1.007`、A-D=`0.223`，原始 H3 为 A=`+1.077`、D=`-1.602`、A-D=`2.679`。因此视觉稳定性恢复不等于 action control 恢复；提交包明确保留这个失败结果。

## 实验协议

正式短片动作诊断统一为：

| 参数 | 值 |
|---|---:|
| RGB frames | 39 |
| latent frames | 12 |
| causal chunks | 3 |
| latent frames/chunk | 5 |
| history chunks | 5 |
| solver steps/chunk | 8 |
| flow shift | 2.22 |
| history | generated |
| anchor | dynamic_last_frame_rgb_dual |
| action routing | causal prefix + feedback |
| KV | persistent raw KV on CPU |
| seed | 13 |
| action gate | flow(A)>0, flow(D)<0, A-D>1.0 |

正式长片使用 124 frames / 8 chunks / 8 steps per chunk，约 5.17 s。

## 关键结果

| Variant | flow(A) | flow(D) | A-D | 视觉结构 | Gate |
|---|---:|---:|---:|---|---|
| Original H3 30-step teacher | +1.181 | -0.842 | 2.023 | stable | PASS reference |
| RGB visual baseline | -1.1535 | -1.4557 | 0.30 | stable after RGB adapter | FAIL |
| RGB Stage2-lite v2 | -1.1369 | -1.4556 | 0.3187 | stable to frame 38 | FAIL |
| Tail4 action-QKV update 1 | -0.8563 | -0.7790 | -0.0774 | stable | FAIL |
| Tail4 action-QKV update 4 | -0.8506 | -0.7684 | -0.0822 | stable | FAIL |
| Tail8 action-prefix update 1 | -1.1091 | -1.4603 | 0.3513 | stable | FAIL |
| Released H3 action-LoRA tail8 | -1.1249 | -1.4151 | 0.2902 | stable | FAIL |
| Corrected own-history endpoint + paired QKV | -0.8086 | -0.6893 | -0.1194 | stable | FAIL |

flow 是 signed horizontal optical-flow proxy，不是完整视频质量指标。它用于检查 A/D 的方向性；视觉质量同时通过解码、contact sheet、人物/停车场结构和 frame-to-frame MAD 检查。

## 视觉稳定性突破

旧 latent-only anchor 的 Stage2-lite 视频在后段出现人物透明和分解。根因不是视频编码，而是训练/推理 anchor protocol mismatch。改为 RGB-consistent dual anchor 后：

- 39 帧 A/D：人物和停车场结构保持到 frame 38；
- 124 帧 W/A/D：人物和场景保持到 frame 123；
- W 长片一次运行约 709 s sampling、31,570 MiB allocated GPU peak、约 13.19 GiB CPU raw-KV history；
- 所有视频为 H.264/YUV420P，完整解码。

对应视频在 videos/visual_stability/，详细报告在 reports/visual_drift_repair/。

## Causal/KV 正确性

124-frame causal rollout 的结构是：

    8 chunks x 8 noisy denoiser forwards = 64 noisy forwards
    8 clean KV commits
    persistent raw KV on CPU
    replay error = 0

这证明 cache 是可复用的 causal history，而不是只做了一个静态 attention mask。单张 L40 可以运行 39 帧 Stage2-lite；训练/诊断的显存峰值约 32–40 GiB，CPU raw KV 约 6.33 GiB/缓存，长片 history window 下约 13.19 GiB。

## Stage2-lite

Stage2-lite 共享一个 frozen H3 backbone，轮换三个小 adapter：student causal action residual、critic/fake-score adapter、teacher（禁用可训练 adapter）。一轮 integrated RGB endpoint run 使用 A/D self-rollout、chunk 1/2、四个 sigma 和 tail4 fake-score adapter，完成无 NaN/OOM；39 帧输出保持人物和场景结构，但 action gate 为 flow(A)=-1.1419、flow(D)=-1.4550、A-D=0.3131。

因此 Stage2-lite 的成功标准是“训练链路能运行并覆盖 generated-history 分布”，不是“已经得到有效的少步 action model”。

## Action geometry 诊断

在同一个 generated state 上：

    frozen causal A/D delta norm = 7.642
    original teacher delta norm = 8.704
    norm ratio = 0.878
    cosine = -0.015

action feedback edge 确实影响 score，但只改变幅度，不能把方向恢复到原始 H3 的 image-space geometry。action_prefix_mode=all 也没有帮助，说明未来 action visibility 不是主要原因。正确的下一步应是 multi-state/multi-seed action supervision，或真正覆盖 student rollout distribution 的 Stage2 训练；继续 single-state gain/anchor/solver sweep 没有清晰归因价值。

## 视频编码检查

本包收录的视频都用 PyAV 批量检查：可打开、帧数完整、24 fps、H.264、YUV420P。旧实验中无法播放或使用 latent-only anchor 的缓存文件不收录为主证据。
