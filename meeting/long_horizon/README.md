# 10/20 秒原始 H3 与 RGB causal 的完整对照

**结果：20 秒 causal 视觉稳定性失败，10 秒后段也有明显退化。** 这组完整视频用于展示当前 checkpoint 的真实长度上限和漂移，不是长视频稳定性已解决的证据。20 秒样本约 10 秒开始严重雾化，15 秒后人物和场景难以辨认；后段没有裁剪。

两组都是从头真实采样的完整长视频，不使用循环、补帧或慢放。固定 W、初始 RGB、场景 prompt、seed=13、flow shift=2.22、832×480 和 24 fps；同长度两边的首帧、prompt、initial video/audio noise 的 SHA256 已验证一致。

## 播放入口

- [10 秒并排对比，243 帧 / 10.125 s](original_vs_rgb_visual_W_10s_243f.mp4)
- [20 秒并排对比，481 帧 / 20.042 s](original_vs_rgb_visual_W_20s_481f.mp4)
- [10 秒抽帧](contact_sheet_10s_comparison.jpg) / [20 秒抽帧](contact_sheet_20s_comparison.jpg)
- [10 秒 causal 原片](causal_W_10s_243f.mp4) / [20 秒 causal 原片](causal_W_20s_481f.mp4)
- [10 秒原始 H3 原片](original_W_10s_243f.mp4) / [20 秒原始 H3 原片](original_W_20s_481f.mp4)

左侧原始 H3 使用 30 full-horizon denoiser calls；右侧保持之前的 RGB visual checkpoint，未重新训练：tail16 visual QKV + action residual、RGB decode/re-encode dual anchor、generated history、causal action prefix、action feedback、CPU raw KV、5 latent frames/chunk、5-chunk history、8 steps/chunk。

243 帧对应 72 latent frames、15 chunks、120 noisy forwards + 15 clean commits；481 帧对应 142 latent frames、29 chunks、232 noisy forwards + 29 clean commits。**8 steps/chunk 不等于全视频只调用 8 次网络。** 不同长度的 noise 张量形状不同，不声称 10 秒与 20 秒的前缀完全相同。

## 单次性能记录

| Length | Method | Noisy + commit | E2E s | Sampling s | GPU peak GiB | CPU KV GiB | First chunk s | Mean chunk s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 10.125s / 243f | original | 30 + 0 | 921.0 | 867.6 | 16.04 | 0.00 | — | — |
| 10.125s / 243f | causal | 120 + 15 | 1773.8 | 1714.1 | 17.87 | 13.19 | 75.3 | 114.3 |
| 20.042s / 481f | original | 30 + 0 | 2406.0 | 2328.7 | 12.10 | 0.00 | — | — |
| 20.042s / 481f | causal | 232 + 29 | 3659.3 | 3562.6 | 30.17 | 13.19 | 57.3 | 122.8 |

GiB = MiB / 1024。GPU 值是每个新进程的 torch.cuda.max_memory_allocated，不是整张卡占用，也不是全部 33B 权重大小。CPU 上的模型权重未计入 CPU KV。首块耗时只度量内部 latent 生成/commit；当前 benchmark 末尾统一解码，不能宣传为真实首屏显示延迟。

这些都是单次运行，没有 warmup 后多次均值。任务并行运行、与其他进程共用主机，且显存 reserve 不同：10s causal/original 分别为 18/20 GiB；20s causal/original 分别为 14/24 GiB。不同 offload 配置会改变速度与显存，因此不据此给出严格加速或显存节省排名。全部 launch/setup JSON 保存在 `source_metrics/`。

## 连续性和描述性运动指标

| Length | Method | Mean RGB MAD | Boundary RGB MAD | Mean horizontal flow |
|---|---|---:|---:|---:|
| 243f | original | 3.802 | 3.876 | +0.486 |
| 243f | causal | 2.713 | 3.642 | -0.692 |
| 481f | original | 3.252 | 3.022 | -0.194 |
| 481f | causal | 2.759 | 4.666 | -0.380 |

边界 MAD 是 0-based frames 17、34、…处相邻 RGB 帧差的均值；水平光流在 416×240 中央裁剪区计算。固定 W 时，水平光流符号不能严格判定前后动作。较低 MAD 也可能来自模糊或运动减弱，不能当作质量提升。

## 完整后段的视觉观察

10 秒：causal 人物轮廓保持到末尾，但约 7–10 秒模糊、ghosting、停车场几何和运动路线漂移更明显，末尾背景变成近处墙面。原始 H3 的停车场细节也会形变。这能展示完整长时 rollout 及其局限，不能写成 10 秒视觉质量或物理一致性已通过。

20 秒：**视觉稳定性明确失败。** 0–5 秒仍能辨认人物和车库；约 10 秒开始严重雾化和重影，15 秒后人物与场景结构难以辨认，20 秒末尾退化成模糊色块。原始 H3 的长时场景几何同样形变，但人物保持得明显更完整。这里没有裁掉失败后段；该视频应作为 generated-history 长时退化的负结果展示，不能作为“20 秒稳定性已解决”的证据。所谓稳定版只指其 124 帧表现相对旧 fixed-mix 较好，不是长时质量保证。

原始 H3 也是生成对照，不是 ground truth。每个长度只有单场景、W、seed=13 的一次运行；不能外推为多场景/多动作长期稳定。A/D 控制并未因为延长时长恢复，动作/视觉取舍应配合 [三列 A/D demo](../action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4) 展示。

## 工程结果及限制

CPU raw KV 在两种长度都达到约 13.19 GiB 后保持 5-chunk 窗口大小。RGB anchor 仍反复解码已生成 prefix，因此当前实现不保证每块耗时恒定。这次没有为长视频修改 checkpoint、action 路径或 anchor。

原始 20 秒 H3 的 eager directed mask 构造曾申请 25.94 GiB 临时张量并 OOM。使用更保守的 GPU watermark 及 `H3_COMPILE_BLOCK_MASK=1` 编译同一 mask 构造后，完整 481-frame baseline 成功运行。mask predicate 没有改变；含 padding 和变更 action assignment 的 dense mask 与 BlockMask metadata 逐元素一致，见 [等价检查](source_metrics/mask_equivalence.json)。失败记录保留在 `source_metrics/original_481_eager_mask_oom/`。

因此不能继续用以前的 baseline OOM 声称原始 H3 天生无法生成长片；这是一项实现/显存配置限制。causal 路径证明了逐块执行和有界历史 KV，未证明相同条件下更快，也未解决生成历史漂移。

## 复现与审计

- `comparison_243_spec.json` / `comparison_481_spec.json`：原始输入路径和 panel labels。
- `../render_comparison.py`：严格匹配分辨率、fps、完整帧数，不裁短或补齐。
- `../summarize_long_horizon.py`：原片 RGB MAD、块边界、Farneback、按约 5 秒分段、contact sheet、noise hash 一致性检查。
- `METRICS.csv` 和 `source_metrics/summary.json`：统一测量。
- 源实验：`H3-World/outputs/2026-10-07-00/long_rgb_visual/`。
- [源代码和小 adapter 指纹](source_metrics/source_fingerprints.json)。
