## 2026-10-07：保留动作较强旧版，完成 10/20 秒长视频对照

按用户要求同时保留 fixed-mix 旧版与 RGB visual 稳定版，新增三列原始/旧 causal/稳定 causal 的 124 帧视频，路径为 `submission/meeting/action_vs_stability/`。旧版 action adapter 已独立备份到 `submission/checkpoints/legacy_fixed_mix/`；A=+0.1427、D=-0.3105、A-D=0.4532，方向符号较好但后段明显漂移。新稳定版 A=-0.7841、D=-1.0075、A-D=0.2233，结构较稳但动作失真。两套模型都保留，不把任何一套写成四方向完全保真。

长视频使用同一稳定 adapter、RGB dual、generated history、CPU raw KV、5 latent frames/chunk、5-chunk history、8 steps/chunk、W、seed=13、shift=2.22、832×480。实际长度为 243 帧/10.125 秒和 481 帧/20.042 秒，分别有 15/29 chunks、120/232 noisy forwards、15/29 clean commits。原始 H3 同长度 30-step 对照同时运行；同长度两分支的首帧、prompt、初始 video/audio noise SHA256 已验证一致。实验目录 `H3-World/outputs/2026-10-07-00/long_rgb_visual/`。

20 秒原始 H3 的 eager directed mask 构造申请 25.94 GiB 临时张量 OOM；保留失败日志，增加可选 `H3_COMPILE_BLOCK_MASK=1` 编译同一 mask 构造后重启。该优化不改变 attention predicate；2176 rows、padding、3 action rows、改变 action assignment 后的 dense mask 与 sparse BlockMask metadata 均逐元素完全一致。补丁与检查脚本归入 submission，未触碰 causal 模型架构。

长视频全部完成并通过完整 H.264/YUV420P、24 fps 解码检查；每个长度 original/causal 的输入指纹一致。真实单次结果如下：

| 帧数 | 方法 | E2E s | GPU peak MiB | CPU KV MiB | Noisy+commit | RGB MAD | Boundary MAD |
|---|---|---:|---:|---:|---:|---:|---:|
| 243 | original | 921.0 | 16425.8 | 0.0 | 30+0 | 3.802 | 3.876 |
| 243 | causal | 1773.8 | 18300.4 | 13508.6 | 120+15 | 2.713 | 3.642 |
| 481 | original | 2406.0 | 12394.8 | 0.0 | 30+0 | 3.252 | 3.022 |
| 481 | causal | 3659.3 | 30896.8 | 13508.6 | 232+29 | 2.759 | 4.666 |

10 秒后段存在模糊、重影和靠墙方向漂移；20 秒：**视觉稳定性明确失败。** 0–5 秒仍能辨认人物和车库；约 10 秒开始严重雾化和重影，15 秒后人物与场景结构难以辨认，20 秒末尾退化成模糊色块。原始 H3 的长时场景几何同样形变，但人物保持得明显更完整。这里没有裁掉失败后段；该视频应作为 generated-history 长时退化的负结果展示，不能作为“20 秒稳定性已解决”的证据。所谓稳定版只指其 124 帧表现相对旧 fixed-mix 较好，不是长时质量保证。

完整材料位于 `submission/meeting/long_horizon/`，三列动作/视觉对照位于 `submission/meeting/action_vs_stability/`；额外备份见 `submission/breakthrough/06_long_rollout_and_tradeoff/`。两边都保留完整后段，不声称动作控制已恢复或长视频质量 PASS。原始 H3 20 秒可通过编译 mask 与 CPU offload 运行，旧 OOM 不能用作原始模型的固有长度限制。单次并发时间和不同 reserve 不能作严格 speedup/memory 排名。


## 2026-10-07：会议视频主结果纠错与视觉协议更换

发现会议包第一版把 `outputs/2026-10-03-02/final_fixed_mix124_8step/` 作为主展示。该旧 fixed-mix 实验只有 action adapter，没有 visual tail16 QKV adapter，推理使用 `dynamic_last_frame_dual` / `global_retimed_latent_dual_v1` latent-only anchor，且 `action_prefix_mode=own`、`action_feedback=false`。它的 MP4 可解码，但 generated-history 后段出现人物透明、车库 tearing 和 temporal-VAE ghosting；“能播放”被错误地当成了“视觉合格”。这是选片/标注错误，不是播放器或编码器故障。旧视频、旧指标和审计已归档到 `submission/meeting/diagnostics/legacy_fixed_mix/`。

会议主入口已改为 `submission/meeting/annotated/h3world_rgb_stable_action_grid_124_timed.mp4` 及同目录 W/S/A/D 单动作视频。它们来自 `outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/`，配置为 tail16 visual QKV + RGB-consistent dual anchor（generated tail 解码 RGB 后经过 H3 image branch）+ generated history + causal action prefix + action feedback + CPU raw KV。W/S/A/D 均为 124 帧、8 chunks、8 steps/chunk、64 noisy forwards + 8 clean commits，已完整解码为 H.264/YUV420P/24 fps。视觉结构比旧主视频稳定，但仍有 blur/ghosting，且 A/D action geometry 没有恢复。

当前 RGB visual-main 单次记录：原始 H3 30-step 为 441.5–454.2 s；causal 为 673.4–767.9 s；首块 41.8–57.4 s；CPU raw KV 13,508.6 MiB（约 13.19 GiB）；causal GPU 峰值 31,569.6–39,939.7 MiB（约 30.8–39.0 GiB）。由于 causal 使用 64 noisy forwards 且带 RGB anchor decode/re-encode，不宣称端到端加速。Farneback flow：原始 A=`+1.077`、D=`-1.602`、A-D=`2.679`；RGB-main A=`-0.784`、D=`-1.007`、A-D=`0.223`，严格 `A>0,D<0,A-D>1.0` gate 失败。

会议文档 `meeting/README.md`、`METRICS.md/csv`、`FAIRNESS.md`、`SLIDES.md`、`MEETING_SCRIPT.md` 已统一到 RGB visual-main；`meeting/source_metrics/rgb_visual/` 保存原始 JSON/flow/continuity。最终结论仍是：causal/KV 工程可行，RGB anchor 能修复主要视觉分解，但 generated-history 下 action-conditioned score geometry 尚未恢复，Stage2-lite 也不等于官方 SGF/DMD。

# GWM 项目进展

更新时间：2026-10-06


## 2026-10-06 16 时段：原始 H3 endpoint latent 监督的 action-QKV smoke 也未恢复动作

在 routing/still 诊断之后，新增了一个不同于单纯 score-delta 的监督来源：对 tail4 action-QKV
使用 full-attention H3 A/D counterfactual paired loss，并在 generated chunks 1、2 的四个
solver sigma 上加入原始 H3 30-step `baseline_latents.pt` 的 chunk endpoint `x0` MSE（权重
0.5）。视觉 tail16 RGB adapter、RGB dual anchor、generated history、CPU raw KV、8 steps/chunk、
seed 13 均保持不变；只做 1 个 optimizer update。

代码新增 `stage2_lite_dmd.py --latent-target-weight`，会从 A/D teacher dirs 读取并校验
`baseline_latents.pt`，因此目标是同一 initial image/prompt/noise 下的原始 H3 endpoint，而不
是新生成的伪标签。实验目录：

```text
H3-World/outputs/2026-10-06-16/action_qkv_latent_endpoint_pair_rgb_39_8step_1update/
```

结果：

| 配置 | flow(A) | flow(D) | A-D | latent A/D delta cosine |
|---|---:|---:|---:|---:|
| 原始 H3 30-step teacher | +1.181 | −0.842 | 2.023 | 1.000 |
| endpoint-QKV causal student | −0.8041 | −0.7047 | **−0.0994** | 0.0563 |

此前没有 endpoint target 的 RGB causal latent delta cosine 为约 `0.0160`；新目标只把它提高到
`0.0563`，仍接近正交，且图像空间 A/D 分离反而变差。因此“直接拟合原始 H3 chunk endpoint”
也不足以恢复 generated-history causal action geometry，不能继续扩大 update 或升级为正式
checkpoint。39 帧视频人物仍可解码，视觉稳定性没有重新出现早期 latent-anchor 的分解，但动作
gate `flow(A)>0, flow(D)<0, A-D>1.0` 明确失败。

可播放对照和报告：

```text
outputs/stage2_action_qkv_latent_endpoint_AD_39.mp4
outputs/2026-10-06-16/action_qkv_latent_endpoint_pair_rgb_39_8step_1update/ACTION_QKV_ENDPOINT_REPORT.md
```

至此，已用 routing upper bound、STILL counterfactual、score-field paired QKV、action-prefix
residual、released H3 action-LoRA、Stage2-lite DMD 和原始 endpoint latent supervision 分别
排除低层修补路径。后续若要再做，必须转为多状态/多 seed 的 rollout-distribution training 或
重新设计 causal action-token/attention topology；不再继续单场景单 state 的 adapter、gain、
anchor 或 endpoint 权重扫描。

需要修正上一段结论的协议边界：上一轮 endpoint paired 分支从同一个 A-generated state 计算
A/D score，却把独立 D teacher rollout 的 endpoint 当成 D target，存在 state mismatch。因此它
只能作为 naive mixed-target 负诊断，不能证明所有 endpoint supervision 都无效。随后新增
`--own-latent-target-weight`，让 A endpoint 只监督 A-generated history，D endpoint 只监督
D-generated history，同时保留 shared-state paired score loss。

修正后的实验目录为：

```text
H3-World/outputs/2026-10-06-16/action_qkv_own_endpoint_pair_rgb_39_8step_1update/
```

训练耗时 818.6 s，GPU peak 约 39.4 GiB，CPU KV peak 约 5.28 GiB，无 NaN/OOM。39-frame 自由
rollout 为 A=`-0.8086`、D=`-0.6893`、A-D=`-0.1194`，latent delta cosine=`0.0655`，仍未通过
`flow(A)>0, flow(D)<0, A-D>1.0`。这排除了同动作 own-history endpoint + 单场景 tail4 QKV 的
1-update 修复，但仍不替代多状态/multi-seed 或完整 rollout-distribution training 的结论。
可播放对照为 `outputs/stage2_action_qkv_own_endpoint_AD_39.mp4`，详细报告为
`outputs/2026-10-06-16/action_qkv_own_endpoint_pair_rgb_39_8step_1update/ACTION_QKV_OWN_ENDPOINT_REPORT.md`。


## 2026-10-06 16 时段：routing 上界与 still 反事实基线完成，action geometry 仍未恢复

在不改动 RGB 视觉适配器、action residual、KV、solver、seed 或视频协议的条件下，补做了一个
只改变 action-prefix 可见性的 routing 上界实验。固定协议为 39 RGB / 12 latent / 3 chunks、
5 latent frames/chunk、8 steps/chunk、flow shift 2.22、generated history、persistent CPU raw KV、
`dynamic_last_frame_rgb_dual`、action feedback、seed 13；视觉 checkpoint 使用
`outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/`。本轮 `action_prefix_mode=all`，
表示对一段已经预先知道的固定动作暴露所有 action rows；它只是 topology upper-bound diagnostic，
不是最终 interactive causal protocol。

实验目录：

```text
H3-World/outputs/2026-10-06-16/routing_all_rgb_39_8step/
```

`all` routing 的 A/D 结果为：

| prefix policy | flow(A) | flow(D) | A-D | gate |
|---|---:|---:|---:|:---|
| all + feedback | -1.1535 | -1.4733 | 0.3198 | FAIL |

为了去除场景本身的共同漂移，又用同一协议生成 `STILL` 反事实：

| action | horizontal flow | 相对 STILL |
|---|---:|---:|
| STILL | -0.7129 | 0 |
| A | -1.1535 | -0.4406 |
| D | -1.4733 | -0.7604 |

因此两点都已排除：

1. 把 action rows 全部提前暴露只把 A-D 从 causal routing 的约 0.31 提到约 0.32，不能恢复
   左右方向；问题不是简单的 future action row 不可见。
2. STILL 反事实不能解释失败。相对 STILL 后 A、D 仍然都向同一负方向偏移，A/D 分离只有
   0.3198，说明共同相机/场景运动不是主要原因。

可播放的 A/D 对照：

```text
outputs/stage2_routing_all_rgb_AD_39.mp4
```

机器可读指标和完整协议见：

```text
outputs/2026-10-06-16/routing_all_rgb_39_8step/ROUTING_STILL_REPORT.md
outputs/2026-10-06-16/routing_all_rgb_39_8step/ROUTING_STILL_REPORT.json
outputs/2026-10-06-16/routing_all_rgb_39_8step/flow_with_still.json
```

本轮仍然没有产生新的正式 124-frame action grid。结合此前 frozen causal/teacher delta cosine
约 -0.015、student/teacher delta norm ratio 约 6.9x，以及 tail4 QKV 4-update learning
curve 未通过图像动作 gate，当前最合理结论是 action-conditioned score-field / causal routing
representation mismatch，而不是缺少一条 action edge 或单纯的 flow baseline 偏置。RGB anchor
修复的人物稳定性结论保持不变，但不能写成 action control 已保留。


## 2026-10-06：RGB anchor + generated-history endpoint adaptation 修复人物分解并通过 124 帧视觉验收

针对早期 `stage2_lite_39_8step_4updates` 视频中人物在后半段透明、分崩离析的问题，先确认了
根因不是 MP4 编码，而是 anchor protocol mismatch：旧 Stage2-lite 将上一 chunk 的 latent
tail 直接 patchify，而 tail16 causal visual adapter 的训练协议是 H3 原生 RGB image branch。

本轮在固定 causal mask、chunk、KV 和 solver 的条件下，只训练 tail16 causal visual QKV，并加入
原始 H3 30-step `baseline_latents.pt` 的 chunk endpoint MSE。协议为：39 RGB / 12 latent / 3
chunks，5 latent frames/chunk，history window=5，8 steps/chunk，flow shift=2.22，generated
history，persistent raw KV on CPU，`dynamic_last_frame_rgb_dual`，causal action rows + feedback，
seed=13；fixed-mix action residual 保持冻结。两个 optimizer update 依次使用 A、D，teacher replay
覆盖每个实际 solver sigma，endpoint loss 权重 0.5，boundary loss 权重 0.25。

实验目录：

```text
H3-World/outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/
```

训练完成且无 NaN/OOM，峰值 allocated GPU 32,509 MiB；每份 student/teacher raw KV 约 6.33 GiB
CPU。A update 用时 553.6 s，D update 用时 522.5 s。39 帧评估均为 H.264/YUV420P，PyAV 完整
解码 39 帧，人物和车库结构到 frame 38 仍存在：

| action | horizontal flow | vertical flow | A-D | visual status |
|---:|---:|---:|---:|:---|
| A | -1.1448 | +0.3306 | — | stable through frame 38 |
| D | -1.4557 | +0.2567 | 0.3109 | stable through frame 38 |

这轮没有恢复 action gate（`flow(A)>0, flow(D)<0, A-D>1.0`），因此不能把它称为 action
geometry 修复；它明确证明 RGB-consistent anchor + endpoint regularization 能解决早期人物分解，
同时没有改变当前 action geometry 的结构性问题。

随后用同一 checkpoint 做 124-frame W 长时 rollout：

- 124 frames / 5.17 s，8 chunks；64 noisy denoiser forwards + 8 clean commits；
- generated history，CPU KV peak 13.19 GiB，GPU allocated peak 31,570 MiB；
- sampling 709.0 s，conditioning 后总耗时 752.9 s；
- H.264/YUV420P，PyAV 完整解码 124 帧；
- 抽帧 0、12、24、38、51、64、76、89、102、111、123 中人物和场景持续完整，没有此前
  latent-only Stage2 的后段分解。

随后在 GPU1/2 并行完成了同协议的 124-frame A、D rollout。两条视频也都完整解码 124 帧，
抽帧至 frame 123 仍保持人物和车库结构：

| action | horizontal flow | gray MAD | visual status |
|---:|---:|---:|:---|
| A | -0.7841 | 2.9972 | stable through frame 123 |
| D | -1.0075 | 2.8282 | stable through frame 123 |

A/D 的负水平 flow 仍是动作几何未恢复的证据，不是人物分解；因此这组长视频只用于视觉稳定
验收，不改变 action gate 的失败结论。新增并排视频为：

```text
outputs/stage2_rgb_endpoint_visual_stable_AD_124.mp4
```

可播放交付物已单独放在 `H3-World/outputs/` 根目录：

```text
outputs/stage2_rgb_endpoint_visual_stable_W_124.mp4
outputs/stage2_rgb_endpoint_vs_original_W_124.mp4
outputs/stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4
outputs/stage2_rgb_endpoint_visual_stable_AD_124.mp4
```

其中第一项是新的 124 帧 causal W，第二项是原始 H3 30-step W 与新 causal W 的并排视频，第三项
是旧 latent-only Stage2 与 RGB/endpoint 版本的 A/D 四宫格对照。详细协议、指标和限制见：

```text
H3-World/outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/VISUAL_DRIFT_REPAIR_REPORT.md
```

当前阶段的结论更新为：视觉 drift/人物分解已经有可复现的 RGB-anchor 修复，并在 124 帧上通过
结构稳定性检查；A/D action geometry 仍未通过 gate。后续若继续，应该把 RGB visual adapter 作为
稳定基线，专门处理 action-pathway 或真正的 rollout-distribution matching，不再混淆视觉修复与
动作控制结果。

随后把该新 visual adapter 接回真正的 `stage2_lite_dmd.py`，完成一轮完整 critic/DMD 集成：
A/D self-rollout，target chunks 1、2，四个 sigmas `0.94,0.79,0.57,0.24`，tail4 critic，
shared 33B backbone，RGB anchor。训练耗时 1111.6 s，GPU allocated peak 40,320 MiB，所有
critic/DMD 梯度有限；实验目录为：

```text
H3-World/outputs/2026-10-06-09/stage2_lite_rgb_endpoint_integrated_39_8step_chunks12_sigmas4_1update/
```

用保存的 student adapter 做同协议 39-frame A/D 验收：

| action | horizontal flow | vertical flow | A-D | visual status |
|---:|---:|---:|---:|:---|
| A | -1.1419 | +0.3277 | — | stable through frame 38 |
| D | -1.4550 | +0.2577 | 0.3131 | stable through frame 38 |

这证明真正的 Stage2-lite critic/DMD 链路不会重新触发人物分解，但单轮 DMD 仍没有恢复 action
geometry。可播放的集成 A/D 对照为：

```text
outputs/stage2_lite_rgb_endpoint_integrated_AD_39.mp4
```


## 2026-10-06 07 时段：Stage2-lite v2 已切换到 RGB-consistent multi-chunk/multi-sigma

为解决此前 Stage2-lite 中“visual tail16 adapter 用 RGB dual 训练、Stage2 rollout 却用 latent-only
dual anchor”造成的人物分解，本轮修改了
[`H3-World/code/causal/stage2_lite_dmd.py`](H3-World/code/causal/stage2_lite_dmd.py)：新增
`--anchor-mode latent|rgb`，默认 `rgb`。当使用 RGB 模式时，student self-rollout、critic
fake-score、frozen teacher 的 history cache 都从同一 generated prefix 解码最后 RGB 帧，再经 H3
image branch `process_image=True` 重编码为第二个 dual anchor；同一 rollout 内按 chunk 缓存，避免
同一边界重复 VAE 转换。latent 模式仍保留作便宜的历史对照。

新增实验目录：

```text
H3-World/outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/
```

固定协议为 39 RGB frames / 12 latent frames / 3 chunks / chunk size 5 / history window 5 /
8 steps per chunk / generated history / CPU raw KV / seed 13；critic 覆盖 chunk 1、2 和
`sigma={0.94,0.79,0.57,0.24}`。仍然是共享一个 33B backbone 的 Stage2-lite feasibility
diagnostic，student hidden action residual 和 tail4 critic 分别只有约 0.77M/0.39M 可训练参数。

训练完成且没有 OOM/NaN：耗时 **872.9 s**，峰值 allocated GPU **39.37 GiB**，最大记录的 CPU
raw-KV cache **5.28 GiB**。A/D 每条评估使用 24 次 noisy denoiser forward + 3 次 clean commit；
RGB anchor 在 chunk 边界增加 VAE decode/re-encode 成本。训练曲线和完整指标见
[`REPORT.md`](H3-World/outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/REPORT.md)。

| 39-frame RGB-dual Stage2-lite v2 | flow(A) | flow(D) | A−D | Gate |
|---|---:|---:|---:|---|
| 当前 checkpoint | −1.1369 | −1.4556 | 0.3187 | FAIL |

本轮的结论分成两部分：

1. **视觉稳定性改善。** A/D 抽帧中人物和停车场结构保留到第 38 帧，之前 latent-only Stage2
   样例中的人物透明/分解没有再出现。可播放 H.264 Constrained Baseline / YUV420P 对照为
   [`AD_rgb.mp4`](H3-World/outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/AD_rgb.mp4)，
   抽帧为
   [`contact_sheet_rgb.jpg`](H3-World/outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/contact_sheet_rgb.jpg)。
2. **动作几何没有恢复。** A 与 D 的水平 flow 都为负，A−D 只有 0.319，未达到短片 gate
   `flow(A)>0`、`flow(D)<0`、`A−D>1.0`，也明显低于原始 H3 teacher 的约 2.02。不能因为画面
   稳定就宣称 action control 已保留。共同的 forward/场景运动占据了水平 flow，当前 causal action
   pathway 仍没有复现 H3 的横向 score geometry。

因此 RGB anchor 修复了主要的视觉 conditioning mismatch，但没有解决 student/teacher action
delta 几乎正交的问题。Stage2-lite v2 的 multi-sigma/multi-chunk 覆盖也没有在一轮更新中恢复
A/D；不继续做 gain 或 anchor sweep，也不把本轮扩展为新的 124-frame 正式 grid。下一步应固定
RGB anchor、KV、solver 和 generated-history，做低容量 action-pathway alignment（以 full H3
counterfactual delta 为目标），然后再决定是否值得继续 DMD/SGF。详见
[`REPORT.json`](H3-World/outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/REPORT.json)。

## 2026-10-06 08 时段：最小 action-pathway alignment smoke 已完成

在固定 RGB dual visual protocol 后，新增了只训练 full-attention H3 A/D counterfactual delta 的
alignment smoke，比较 hidden action residual 和零初始化 tail4 action-QKV refiner。两者都使用
39 帧 / 3 chunks / 8 steps per chunk / generated history / CPU raw KV / 四个 sigma / seed 13，
visual tail16 QKV 冻结；不是重新训练 visual adapter，也没有引入新的 anchor 或 solver。

实验目录：

```text
H3-World/outputs/2026-10-06-08/action_align_hidden_rgb_39_8step_pair1_final/
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair1_final/
```

paired loss 只在同一个 generated A state 上匹配 student/teacher 的 A−D score delta。结果如下：

| Variant | paired delta cosine | student/teacher norm ratio | flow(A) | flow(D) | A−D | Gate |
|---|---:|---:|---:|---:|---:|---|
| hidden residual | 0.139 | 1.974 | −1.2499 | −1.4293 | 0.1794 | FAIL |
| tail4 action-QKV | **0.390** | **1.018** | −0.8359 | −0.7685 | −0.0675 | FAIL |

QKV refiner 的 score-field alignment 明显优于 hidden residual，但这一次 alignment 没有转化为
自由 generated-history 视频的图像空间 A/D 方向；两条视频都保留人物和停车场结构到第 38 帧，
所以本轮进一步排除了“只要把 teacher delta 对齐就能自动恢复 action geometry”的假设。可播放
对照和抽帧为：

- [hidden A/D](H3-World/outputs/2026-10-06-08/action_align_hidden_rgb_39_8step_pair1_final/AD.mp4)
- [tail4 QKV A/D](H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair1_final/AD.mp4)
- [contact sheet](H3-World/outputs/2026-10-06-08/action_alignment_contact_sheet.jpg)
- [完整报告](H3-World/outputs/2026-10-06-08/ACTION_ALIGNMENT_REPORT.md)

本轮说明 action representation 仍有更深的 causal routing/topology 问题；不能把 paired loss
下降或 delta cosine 上升直接写成 action control 恢复。已启动一个 tail4 QKV 的 4-update
learning curve，并在脚本中加入每轮 `update_XX/student_action_adapter.pt` 保存，完成后再用
同一 39-frame gate 选择是否继续。

## 最新结论：RGB-dual + FP32 gain 4-step 曲线已完成，39 帧 gate 未通过

当前需要先看这一节。下方保留按实验阶段累积的历史记录，历史中的“正在运行/下一步”不再
代表最新状态。最新完整报告和可播放对照为：

- [本轮报告与效率表](H3-World/outputs/2026-10-06-09/GAIN_CURVE_REPORT.md)
- [原始 H3 / 旧 latent-anchor 失败样例 / 当前 RGB-dual step_01 对照](H3-World/outputs/h3world_rgb_gain_diagnostic_AD_39.mp4)
- [A/D 抽帧图](H3-World/outputs/2026-10-06-09/contact_sheet.jpg)
- [完整训练记录](H3-World/outputs/2026-10-06-09/trainable_gain_fp32_full_teacher_rgb_8x4/training.json)

| 39-frame 配置 | flow(A) | flow(D) | A−D | 数值 gate |
|---|---:|---:|---:|---|
| 原始 H3 30-step reference | +1.181 | −0.842 | 2.023 | 参考 |
| RGB dual 手工 A64/D8 初始化 | +0.0474 | −0.7812 | 0.8286 | FAIL |
| FP32 gain step_01 | +0.0489 | −0.7782 | 0.8272 | FAIL |
| FP32 gain step_02 | +0.0445 | −0.7629 | 0.8075 | FAIL |
| FP32 gain step_03 | +0.0431 | −0.7782 | 0.8213 | FAIL |
| FP32 gain step_04 | +0.0350 | −0.7652 | 0.8002 | FAIL |

本轮在单卡 GPU0 完成 4 次 A/D 交替 optimizer update，GPU1/4 并行回载评估；其它用户
VLLM 进程没有改动。训练耗时 **2265.4 秒（37.8 分钟）**，峰值 allocated 显存
**40.06 GiB**，student / teacher 的 CPU raw KV **各 6.33 GiB**。

训练的是已有 action residual projection 和 9 个 gain（共 3,096,585 参数），不是只训
9 个数；visual tail16 QKV 和原 H3 LoRA 冻结。初始化仍是手工选出的 A64/D8，所以这轮
不能被描述成“从单位增益学会动作幅度”。每条评估保持 39 RGB frames / 12 latent frames /
3 chunks、8 steps/chunk、RGB dual、generated history、seed13；所有 checkpoint 的
initial noise、audio noise、初始 image anchor 和同一 action 的 prompt embedding 都和
保存的 teacher conditioning 逐元素核验。每条视频是 **24 noisy forwards + 3 clean commits**。

本轮修正了一个 gain 数值实现问题：此前 gain 保存为 FP32，但投影前转成 BF16，导致
63.9836/8.0092 在前向中又变成 64/8。现在只有很小的 action projection 用 FP32，完成的
residual 再转 BF16。checkpoint 保存 `gain_compute_dtype`，旧文件没有该字段时保持原计算
路径；新增精度、梯度和 checkpoint 兼容测试，相关测试 **9 passed**。修正前的
`2026-10-06-08/trainable_gain_full_teacher_rgb_8x4` 在 step_01 后主动中断并标为 interrupted。

修正后的 gain 最终为 A≈63.9344、D≈8.0369。虽然同一 rollout action 下 replay/magnitude
loss 有所下降，但 direction loss 仍约 1，说明 teacher/student delta 方向尚未对齐；rollout
的 A−D 没有超过初始化。A 的正向光流很小，不能仅以正负号宣称 action controllability 已恢复。
RGB dual 的视觉改善和动作几何恢复是两个不同结论，MAD/边界 MAD 也不能当视频质量分数。

抽帧复核：新 step_01 / step_04 人物主体保留到第 38 帧，不再像旧 latent-anchor Stage2-lite
样例那样基本消失；但 A 有明显向远处走/人物缩小的趋势，不能视为正确横移，D 末尾腿部仍
模糊。根目录三列对照为展示不同阶段结果，各列 adapter 不完全一致，并非 anchor-only
消融。视频已验证 39/39 帧解码、H.264 Constrained Baseline、YUV420P、24 fps。

推理效率也需如实报告：step_01 A/D 含加载/conditioning 总耗时约 246.0/224.7 秒，峰值
39.17 GiB，CPU KV 6.33 GiB；原始 30-step A/D 是 212.4/218.0 秒、38.99 GiB、无历史 KV。
时间存在并行训练/CPU offload 负载差异，这一短片 RGB-dual 配置没有证明速度或显存优势。

**决定：不继续 gain/anchor 扫描，不生成新的 124-frame 正式 grid。** 这只是一个场景、一个
seed、4 次更新的负结果，不能证明架构上限或证明必须使用 SGF/DMD。进一步训练前应先审计
full-teacher target：当前 teacher 是双向 generated-prefix/current-chunk 前向，并不是原始
完整 39-frame teacher 轨迹；要先确认同一 state 上 teacher delta 的方向、强度和 student
可学习梯度，再决定 action-path adaptation 或 RGB 一致的多 sigma critic/DMD。正式
124-frame 交付仍为旧的 `h3world_final_fixed_mix_action_grid_124.mp4`。

## 2026-10-06：full-teacher counterfactual delta audit

为避免把“关闭 causal mask 的 teacher”直接当成正确动作监督，新增了只读脚本
[`H3-World/code/causal/audit_teacher_action_delta.py`](H3-World/code/causal/audit_teacher_action_delta.py)。
它取 RGB-dual step_01 的 A generated-history latent，在同一个 chunk/current state 上分别
运行原始双向 H3 的 A、D conditioning，扫描 3 个 chunk 和 5 个 solver sigma；不训练任何
参数。原始数据和表格位于：

```text
H3-World/outputs/2026-10-06-10/teacher_delta_audit/audit.json
H3-World/outputs/2026-10-06-10/teacher_delta_audit/REPORT.md
```

teacher 的 A/D velocity 都有限且 delta 不为零，但 delta 相对平均 velocity norm 的比例只有
约 `0.0128–0.0336`，15 个 chunk/sigma 状态均值约 `0.0207`。因此 full teacher 确实给了可反传的动作信号，
但它是完整 score field 中很小的差异；4-step residual/gain 更新只能造成弱的 rollout 改变，
并不能从这个结果推出目标方向与图像空间 strafe 一致。audit 只使用一个 A generated state、
一个场景和一个 seed，也没有测 student 对齐或 optical flow，不能被描述为完整 action accuracy
结论。随后用 D-generated state 复核同一脚本，delta/velocity ratio 为
`0.0150–0.0424`、均值 `0.0206`；A-state 和 D-state 的均值都约 2%，说明“小而非零”的
监督信号不是只由某一个 rollout 方向造成。两组汇总见
[`COMBINED_REPORT.md`](H3-World/outputs/2026-10-06-10/teacher_delta_audit/COMBINED_REPORT.md)。
这仍不证明 target direction 与图像空间 strafe 一致；下一步应记录 student/teacher delta
cosine，再决定扩大 action pathway 还是做 RGB-consistent multi-sigma critic/DMD。

## 2026-10-06：student/teacher action-delta 对齐审计完成

进一步加载 RGB-dual gain curve 的 `step_01` causal student，在同一个 A generated latent、
同一份 detached causal raw-KV history、同一 chunk 和同一 sigma 上计算 student A/D delta，
并和 full-attention teacher delta 直接比较。15 个 chunk/sigma 状态的结果为：

```text
student delta norm mean: 69.70
teacher delta norm mean: 10.68
student / teacher norm ratio: mean 6.89x, median 7.31x, range 4.21x–9.91x
student-teacher delta cosine: mean -0.0087, median -0.0115, range -0.2257–+0.1178
```

原始逐状态数据和复现脚本位于：

```text
H3-World/outputs/2026-10-12/student_teacher_delta_audit.json
H3-World/outputs/2026-10-12/STUDENT_TEACHER_DELTA_REPORT.md
H3-World/code/causal/audit_student_teacher_delta.py
```

这比 teacher-only audit 更明确：teacher delta 虽小但非零，而 causal student 的 A/D score
差异约大 4–10 倍且几乎正交。换句话说，当前动作路径产生的是不同的 action-conditioned
score geometry；只调 gain 不能旋转这个方向，所以继续增加 gain optimizer steps 没有归因
价值。A/D flow 的正负号可以暂时正确，但不能据此声称已经保留原始 H3 的横移几何。下一步
应先适配/审计 action pathway 或 causal attention topology，使 student delta 对齐 teacher，
再考虑 critic/DMD；不应先用 distribution-level loss 掩盖表示/拓扑不匹配。

## 2026-10-05：真正的 SolarWM-style Stage2-lite 已完成首轮验证

本轮开始尝试用户提出的“真正 Stage2-lite”，与之前的 minimal two-pass replay
明确区分。新增脚本
[`H3-World/code/causal/stage2_lite_dmd.py`](H3-World/code/causal/stage2_lite_dmd.py)，在
同一个 H3-World 33B backbone 上轮换三个小 adapter：

```text
student causal action residual
critic/fake-score hidden residual
frozen teacher = backbone with causal/action residual disabled
```

训练链路为：

```text
student self-rollout (generated history, detached)
→ critic 在 student 生成的最后 chunk 上做 flow matching
→ frozen H3 teacher 和 critic 分别预测同一 noisy state
→ fake_x0 - real_x0 构造 DMD gradient
→ student replay 的 surrogate loss 反传到 action residual
```

H3-World 当前 causal DiT 返回的是 `noise-clean` velocity，故本实现使用
`x0 = noisy - sigma * velocity`。SolarWM 独立 SGF backend 在 adapter 后采用 data-ward
符号，不能直接把其正负号照搬到 H3-World。

### Smoke 与正式 39 帧结果

统一协议仍为 39 RGB frames / 12 latent frames / 3 chunks、chunk size 5、history window 5、
`flow_shift=2.22`、dynamic dual latent anchor、CPU raw KV、generated history、causal action
rows、action feedback、seed 13。Stage2-lite 第一版只在最后 generated chunk、`sigma=0.6` 做
critic 和 DMD 更新；critic 使用 tail4 hidden residual，student 与 critic 不复制 33B。

| 实验 | solver | 更新轮数 | flow(A) | flow(D) | A-D | 结论 |
|---|---:|---:|---:|---:|---:|---|
| fixed-mix Stage1 baseline | 8 | 0 | -0.01325 | -0.76891 | 0.75566 | 基线 |
| Stage2-lite smoke | 2 | 1 | -0.01205 | -0.18138 | 0.16933 | solver 不可与正式基线直接比较 |
| Stage2-lite | 8 | 1 | -0.01380 | -0.77279 | 0.75900 | 链路可行，未改善 |
| Stage2-lite | 8 | 4 | +0.00477 | -0.76839 | 0.77316 | A 符号恢复，未过 A-D>1.0 gate |

4 轮正式实验耗时约 1064 秒（约 17.7 分钟）。每个 action 都重新执行 3-chunk student
self-rollout、critic update 和 student DMD update。峰值 allocated GPU memory 约 39.4 GiB；
每个 CPU raw-KV cache 约 5.28 GiB，未加载第二份或第三份 33B backbone。所有 critic loss、
DMD surrogate 和梯度均为有限值，没有 OOM 或 NaN。

结果目录：

```text
H3-World/outputs/2026-10-05-03/stage2_lite_smoke39_2step/
H3-World/outputs/2026-10-05-04/stage2_lite_39_8step_1update/
H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/
```

正式 4 轮的训练曲线、adapter、原始 flow JSON 和结论见
[`RESULTS.md`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/RESULTS.md)、
[`summary.json`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/summary.json) 和
[`stage2_lite.json`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/stage2_lite.json)。
最终 A/D 并排视频为
[`stage2_lite_39_8step_4updates_AD.mp4`](H3-World/outputs/stage2_lite_39_8step_4updates_AD.mp4)。
抽帧 contact sheet
[`contact_sheet.jpg`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/contact_sheet.jpg)
显示人物在约第 15–30 帧逐渐 ghost/雾化，末尾基本消失；因此 DMD 首轮没有解决视觉 drift。

### 播放兼容性修复

2026-10-05 发现 `eval/D/cached.mp4` 在部分播放器中无法播放。文件本身可以由 PyAV 解码
39 帧，但原始视频是 H.264 High Profile；现已在原路径重新编码为 H.264 Constrained Baseline、
YUV420P、24 FPS，并保留原文件为 `eval/D/cached_highprofile.mp4`。A 视频和根目录的并排视频
也统一转成相同的兼容编码；重新检查均为 39/39 帧可解码。现在应直接使用原路径
[`eval/D/cached.mp4`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/eval/D/cached.mp4)
和根目录的
[`stage2_lite_39_8step_4updates_AD.mp4`](H3-World/outputs/stage2_lite_39_8step_4updates_AD.mp4)。

### 当前判断

这次实验已经证明的是：

```text
H3 causal student self-rollout
+ generated-history fake-score critic
+ frozen teacher
+ DMD surrogate
```

可以在现有单卡 46 GiB L40 上以共享 33B backbone 运行并完成真实反传。它还没有证明
Stage2-lite 能恢复 H3 的 action geometry：4 轮后 A-D 只有 0.773，低于短片验收线 1.0，
也远低于原始 H3 teacher 约 2.02。A 仅略高于 0，D 保持负向，因此不能把这个结果称为成功的
Stage2 checkpoint。

当前最合理的解释是训练信号仍然过窄：critic 只看最后 chunk 和单个 sigma，DMD 只对一次
counterfactual state 更新；它更像验证了训练机制和显存方案，而不是足够的 Stage2 distribution
matching。后续若继续，优先扩展到多个 solver sigma 和所有 generated chunks，并保留相同的
student/critic/teacher 共享 backbone；不要再把这轮结果覆盖正式 124-frame 主 demo。

## 2026-10-02 08 时后：为什么 W/S/A/D 还不能完全保真

当前四方向实验的准确结论不是“action control 已经完全恢复”，而是：动作条件仍能改变
causal 输出，但因果注意力、raw-KV 提交方式、teacher-forcing 与自由 rollout 的分布差异
共同削弱了方向响应。四方向的数值证据如下（同一首帧、prompt、seed 和初始噪声）：

| 条件 | A flow | D flow | A-D |
|---|---:|---:|---:|
| 原始 H3 teacher，39 帧，30 steps | +1.181 | -0.842 | +2.023 |
| causal，无 adapter，clean history，39 帧 | +0.391 | -0.565 | +0.956 |
| causal，无 adapter，generated history，39 帧 | -0.137 | -0.152 | +0.015 |
| causal，无 adapter，generated history，124 帧，8 steps/chunk | -0.018 | -0.026 | +0.008 |

这说明问题分成两个阶段。首先，即便历史使用 teacher latent，causal mask 也把 A-D 差异
从约 2.02 降到约 0.96：当前 chunk 看不到未来 video token，prefix/action token 在缓存
路径中不能像原始全序列 H3 那样与当前 video token 双向更新，且 action LoRA 原本是在
全序列、每个噪声时刻的注意力图上训练的。现在提交到 raw-KV 的是 sigma=0 clean chunk，
之后 noisy denoising 读取的是固定历史 K/V，和原始 H3 每一步重新计算所有历史 token 的
score field 不是同一个函数。

其次，free-running rollout 使用模型自己生成的历史。当前 Stage1 adapter 的训练主要是
clean-history teacher forcing，因此每个 chunk 的小误差会被写入下一次的 K/V 和 image
anchor；动作差异是相对于场景、首帧和 prompt 的较小 residual，进入未见过的 generated
history 后很容易被共同的场景先验覆盖。39 帧 generated-history 已经把 A-D 压到约
0.015，124 帧、8 steps/chunk 进一步降到约 0.008。增加 solver steps 不能消除这个训练/推理
分布差异；这正是需要 SolarWM Stage2 式 self-rollout distribution matching 的地方。

普通共享 MSE adapter 还存在目标冲突：四个动作共享 QKV 尾层，最容易降低平均 denoising
loss 的方式是学习共同的外观和运动，而不是保留 A/D 的符号。pair loss、base-output
regularization 和 schedule supervision 能提高动作差异，但没有建立显式的方向几何约束；
per-action adapter 虽能把 A-D 差异提高到约 0.323，却在约 30 帧后出现明显雾化和 temporal
ghosting，不能作为主结果。因此不能把“数值分开”误称为“视频质量和 action fidelity 同时
恢复”。

anchor 也有独立影响。latent dual anchor 只是把 temporal latent patchify 成 image-like
prefix，不等价于 H3 原生的 RGB decode → `process_image=True` encode 路径；RGB dual anchor
能显著改善后段结构连续性，但当前无 adapter 版本的 A/D flow 约为 -0.032/-0.028，方向
几乎没有水平流响应。RGB adapter 又会重新引入后段雾化。当前最佳视觉候选是
`H3-World/outputs/h3world_action_grid_rgbdual_124.mp4`，最佳动作分离候选是
`H3-World/outputs/h3world_action_grid_individual_reg_124.mp4`；二者代表质量与方向响应的
明确 trade-off，均不能宣称四方向完全保真。

最后，水平 optical flow 只适合直接判断 A/D。W/S 主要产生深度、尺度、主体位移和相机跟随
变化，原始 H3 的 W/S 水平 flow 本身也同号，因此不能用“W flow 必须为正、S flow 必须为负”
作为验收条件。W/S 应结合主体尺度/深度变化和视频对照，A/D 才用左右 flow 符号；当前报告
将 action sensitivity、directional fidelity、temporal continuity、visual quality 和
generated-history stability 分开记录。

因此当前阶段的严谨结论是：H3-World 的 action rows、causal chunk rollout 和 persistent
raw-KV 链路已经打通；clean-history 短时 rollout 保留部分动作响应，但 free-running history
会造成动作塌缩。要同时恢复四方向和长时质量，需要把 action-sensitive schedule supervision
与 generated-history self-rollout 训练结合起来，并重新设计能保持 H3 image-condition 语义
的局部 anchor；仅继续堆叠 mask、anchor 或增加 solver steps 不足以证明 Stage2 效果。

### W→A→D 时间切换验证

刚完成 `W:3,A:2,D:3` 的 124 帧 RGB dual-anchor rollout，结果目录为
`H3-World/outputs/2026-10-02-04/action_schedule_WAD_base_rgbdual_124/`，并排视频为
`H3-World/outputs/h3world_schedule_WAD_rgbdual_124.mp4`。因果分支使用 64 次 noisy
denoiser forward、8 次 clean commit、约 13.19 GiB CPU raw-KV、约 38.7 GiB allocated
GPU peak，采样时间约 590.0 s；这是无额外 causal adapter 的视觉连续性候选。

分段 Farneback central-flow proxy（W 0–50 帧、A 51–84 帧、D 85–123 帧）如下：

| 视频 | W flow x / magnitude | A flow x / magnitude | D flow x / magnitude |
|---|---:|---:|---:|
| Original H3 30-step | -0.033 / 1.108 | +1.564 / 2.072 | -1.278 / 2.291 |
| Causal RGB dual, 8-step/chunk | -0.102 / 0.345 | -0.003 / 0.301 | -0.002 / 0.147 |

原始 H3 在 A→D 切换处出现明显的正负水平流变化；RGB dual causal 视频保持结构但没有
复现这个方向切换，进一步验证“视觉连续性”和“action directional fidelity”是两个独立
问题。完整 JSON 为 `action_schedule_WAD_rgbdual_flow.json`。

## 2026-10-01 04–05 时段：修正 Stage1 rollout 调度并复测 anchor

本时段发现并修正了一个会直接造成“只闪两下”的 benchmark 错误：`benchmark.py` 原来在
每个 chunk 内调用了所有 timestep 的 denoiser，却把 `scheduler.step(...)` 放在 timestep
循环外。因此日志虽然记录了 8 次 denoiser forward，latent 实际只前进了一次，不能作为
8-step causal rollout 结果。现在 scheduler 更新已经放回循环内；每个 chunk 的 N steps
都会产生 N 次 latent 更新，chunk 结束后仍额外用 `sigma=0` 做一次 clean KV commit。
之前被终止的 `outputs/2026-10-01-04/fixed_base30_124/` 和
`fixed_base8_shift2_124/` 已标记为 `aborted`，不再作为实验结果。

### 修正版 22 帧 A/B

使用相同 seed 13、相同单末层 Stage1 adapter
`outputs/2026-10-01-01/stage1_long_retimed_train_seed13/adapter.pt`、8 steps/chunk、
shift=12、5-frame chunk。baseline 是同一条件下的原始 H3 30 steps。结果目录为
`H3-World/outputs/2026-10-01-05/`：

| 配置 | 采样时间 | denoiser / clean commit | GPU allocated peak | CPU raw-KV peak | 灰度 MAD 均值/p95 | 空间边缘差 |
|---|---:|---:|---:|---:|---:|---:|
| H3 original, 30 steps | 118.07 s | 30 / 0 | 39,922.7 MiB | 0 | 3.493 / 12 | 2.242 |
| causal, fixed first-frame anchor, 8 steps | 51.11 s | 16 / 2 | 39,923.5 MiB | 3,782.4 MiB | 2.779 / 9 | 2.093 |
| causal, previous-last-frame anchor + W action, 8 steps | 50.76 s | 16 / 2 | 39,923.5 MiB | 3,782.4 MiB | 3.341 / 11 | 2.280 |

修正后 8-step rollout 的输出已经是连续的多步更新，不再是旧 bug 造成的单步跳变。动态
末帧 anchor 的运动量比 fixed anchor 高，也更接近 30-step teacher；但这是 22 帧（约
0.92 秒）的短片，不能据此宣称长片质量已经解决。并排文件为
`H3-World/outputs/stage1_corrected_dynamic_anchor_8step_22.mp4`，fixed/dynamic 严格 A/B
为 `H3-World/outputs/stage1_corrected_fixed_vs_dynamic_22.mp4`。

“上一 chunk 最后一帧 + action”的实现协议是：当前 chunk 继续读取允许的历史 raw K/V，
同时把上一已生成 clean chunk 的最后一个 latent frame 替换到显式 H3 anchor 行，并把
anchor 的 temporal position retime 到真实的全局 frame index；action 文本仍按每个 latent
frame 的 W 按钮条件注入。当前 anchor 是 latent patchify 的最小原型，不等同于先解码 RGB
再用 H3 `process_image=True` 独立编码的原生 image anchor；后者需要单独的局部 anchor
layout 和 denoise mask，仍列为后续改造。

末层固定 anchor 的 4-block QKV LoRA 低学习率复测也已完成：训练 loss
`0.13966→0.11653`、validation `0.11173→0.09901`，但修正版短片 MAD 只有 `2.105/7`，
表现为运动被压平，暂不作为 Stage1 候选。它只保留在
`outputs/2026-10-01-04/stage1_fixed_tail4_lr1e4/` 作为负面 ablation。

### 124 帧修正版长片

同一单末层动态 anchor adapter 的 124 帧、8 steps/chunk rollout 已完成：64 次 denoiser
forward、8 次 clean commit，采样 `299.46 s`，VAE decode `12.73 s`，CPU raw-KV 峰值仍为
约 `13.5 GiB`。与同 seed 的 30-step teacher 比较，teacher 灰度 MAD 为 `4.295/15`、
边缘差 `2.226`，causal 为 `5.735/22`、边缘差 `4.714`。抽帧显示前几个 chunk 能保持人物
运动，但约第 30 帧后出现纹理重影、亮度漂移和场景结构破坏。因此修正后的 Stage1-style
链路可以稳定执行真实 N-step rollout，却仍未通过长片质量验收；根因仍是 clean-history
训练到 generated-history 自回归的分布差异，以及 latent anchor 与 H3 独立图片编码语义不
完全一致。长片并排视频为 `outputs/stage1_corrected_dynamic_anchor_8step_124.mp4`。

同一 adapter 的 16 steps/chunk 长片也已完成，用于区分积分步数误差和训练分布误差：采样
`586.63 s`，128 次 denoiser、8 次 clean commit；causal 灰度 MAD 为 `6.575/26`、边缘
差 `3.836`，仍明显劣于 30-step teacher。因此简单把 8 增到 16 步不能修复长片漂移，问题
主要在训练/rollout 分布而不是单纯积分误差。结果在
`outputs/2026-10-01-05/stage1_dynamic_corrected_124_steps16/`，并排文件为
`outputs/stage1_corrected_dynamic_anchor_16step_124.mp4`。无论 8 或 16 steps，都不称为
Stage2，因为没有 SGF/DMD student 蒸馏。

为对齐 H3 原生 scheduler，另在 GPU 3 启动了 `shift=2.22`、dynamic anchor、124 帧全部
chunk 的 Stage1-style 末层训练，目录为
`outputs/2026-10-01-05/stage1_dynamic_shift2_train_tail1_gpu3/`；此前 GPU 2 的同配置在
第 7 个最大 packed chunk 因显存临时 mask 分配失败，已保留为 failed 记录，不作结论。

该 shift=2.22 adapter 已完成 124 帧回载，使用完全相同的 8-step scheduler、dynamic
previous-last-frame anchor 和 W action：64 次 denoiser、8 次 clean commit，采样
`324.34 s`，CPU raw-KV 峰值约 `13.5 GiB`。它是当前最好的 Stage1-style 候选：teacher
的灰度 MAD/边缘差为 `4.295/15`、`2.226`，新 causal 为 `4.452/17`、`2.433`；相比
shift=12 adapter 的 `5.735/22`、`4.714` 有明显改善。训练指标为 train
`0.10522→0.08774`、validation `0.10618→0.09551`、replay error `0`。但抽帧仍显示约
第 30 帧后有重影和亮度漂移，所以当前结论是“协议对齐显著改善、长片质量仍未完全通过”，
不是 Stage2 成功。当前主对比视频为
`outputs/h3world_30step_vs_stage1_aligned_shift2_8step_124.mp4`（同内容的时段副本为
`outputs/stage1_stage1_aligned_shift2_dynamic_anchor_8step_124.mp4`），统计为
`outputs/2026-10-01-05/metrics_124_shift2.json`。

## 2026-10-01 Stage1 动态末帧 anchor 与 124 帧扩展

本轮继续围绕面试题主线“在 H3-World 中验证 SolarWM 的因果少步生成思路”，没有转去
完整复现 SolarWM 或下载其大规模数据。重点是把 Stage1 teacher-forcing 的条件协议和
causal rollout 对齐，并验证用户提出的想法：每个新 chunk 除了历史 K/V 外，再显式给模型
上一个 chunk 的最后一帧，作为类似 H3 image-to-video 首帧的视觉 anchor。

### 新增实现

- `H3-World/code/causal/h3_cached.py` 新增 `last_frame_anchor()`，把单个 clean latent
  frame 转成 H3 keyframe anchor rows。
- `H3-World/code/causal/benchmark.py` 新增 `--anchor-mode fixed|dynamic_last_frame`。
  动态模式从 chunk 1 开始使用上一已生成 clean chunk 的最后 frame；重复 anchor 只进入
  当前 prefix，不重复写入 video raw-KV cache。
- `H3-World/code/causal/train_pretrained_multichunk.py` 新增同样的动态 anchor 规则、H3
  shifted scheduler 噪声点、任意 latent 长度/target chunk 列表和训练/验证 noise seed
  参数；同时提高 Dynamo shape-cache 上限，支持长片不同 chunk 的变长 packed sequence。
- 当前动态 anchor 仍替换原首帧的 packed value，使用原首帧 position；这是诊断实现，不是
  最终的局部 anchor layout。正确实现仍需局部 anchor token、重复帧 denoise mask、局部或
  全局 RoPE 位置语义，以及只提交新 rows 的 cache 生命周期。

### 22 帧动态 anchor A/B

原始产物目录：`H3-World/outputs/2026-10-01-01/`。

同一个 seed 13、同一个旧 multi-chunk adapter、8 steps/chunk 的 fixed/dynamic A/B：

| 输出 | 采样时间 | denoiser forwards | clean commits | GPU allocated peak | CPU raw-KV peak |
|---|---:|---:|---:|---:|---:|
| fixed first-frame anchor | 75.80 s | 16 | 2 | 34,702.6 MiB | 3,782.4 MiB |
| dynamic previous-last-frame anchor | 66.96 s | 16 | 2 | 34,702.6 MiB | 3,782.4 MiB |

短片描述性统计：30-step teacher 的灰度帧间 MAD 均值/p95 为 `3.490/12`、空间边缘差
`2.242`；fixed 为 `2.425/11`、`2.277`；dynamic 为 `2.573/11`、`2.402`。动态 anchor
确实减少完全静止倾向，但也出现亮度/场景漂移；这些统计不是画质分数。直接 A/B 视频为
`H3-World/outputs/h3world_fixed_vs_dynamic_anchor8_22.mp4`。

### Stage1 训练与短片回载

使用 seed 13、22 帧 teacher、chunk 0/1、末层 rank-8 QKV LoRA、动态 anchor、H3 8 点
shift-12 schedule 训练 160 步：train `0.0862→0.0566`，validation `0.0548→0.0461`，
训练样本 16、验证样本 4、replay 最大误差 0，可训练参数 215,040。adapter 位于
`H3-World/outputs/2026-10-01-01/stage1_dynamic_anchor_seed13/adapter.pt`。

回载到 22 帧动态 anchor、8 steps/chunk 后：采样 `62.41 s`，GPU allocated peak
`34,702.6 MiB`，CPU raw-KV peak `3,782.4 MiB`，灰度帧间 MAD `3.020/14`，空间边缘差
`2.297`。它比旧 adapter 更有运动，但不足以证明质量等价。并排视频为
`H3-World/outputs/h3world_30step_vs_stage1_dynamic_anchor8_22.mp4`。

### 同 seed 124 帧对照

为满足面试题的长视频交付，使用相同 seed 13、首帧、prompt、action、832×480、124 帧
（5.17 秒）重新生成：

- 原始 H3 30 steps：采样 `368.77 s`，产物 `baseline30_seed13_124/baseline.mp4`；
- 22 帧训练 adapter + dynamic anchor 8 steps/chunk：采样 `478.52 s`，64 次 denoiser
  forward、8 次 clean commit，约 `13.5 GiB` CPU raw-KV；约第 40 帧后场景明显漂移。

原始产物在 `outputs/2026-10-01-01/baseline30_seed13_124/` 和
`stage1_dynamic_rollout_seed13_tail8_124/`，并排诊断视频为
`H3-World/outputs/h3world_30step_vs_stage1_dynamic_anchor8_124.mp4`。这证明短片 anchor
实验不能直接外推到长片。

### 124 帧多 chunk Stage1 teacher-forcing

为修复上面的训练/rollout 分布缺口，直接在 124 帧 teacher 的 chunk 0–7 上做 clean-history
训练（单 noise seed、每 chunk 4 个 H3 scheduler sigma、每 chunk 2 个 validation 样本）：

| 项目 | 结果 |
|---|---:|
| train/validation samples | `32 / 16` |
| optimizer steps | `240` |
| train loss | `0.09109 → 0.07251` |
| validation loss | `0.05029 → 0.04664` |
| replay max error | `0` |
| train allocated peak | `6,270.2 MiB` |
| feature extraction / total wall | `460.83 s / 601.35 s` |

adapter：`H3-World/outputs/2026-10-01-01/stage1_long_dynamic_anchor_seed13/adapter.pt`。
回载结果位于 `stage1_long_rollout_seed13_tail8_124/`：采样 `316.83 s`，64 次 denoiser
forward、8 次 clean commit、CPU raw-KV peak `约 13.5 GiB`。长片统计：teacher
灰度 MAD `4.295/15`、边缘差 `2.226`；long Stage1 causal 为 `5.887/23`、`4.343`。
抽帧显示运动量增加，但中段以后出现重影、亮度漂移和场景结构破坏。最终长片并排视频：
`H3-World/outputs/h3world_30step_vs_stage1_long_anchor8_124.mp4`。这是 Stage1 链路可运行
但质量仍失败的结果，不是 Stage2 或质量等价的少步模型。

当前最重要的结论：显式末帧 anchor 对短片运动有帮助，但直接把它放在首帧 packed position
不是长期稳定的实现；clean-history teacher-forcing 即便覆盖全部 8 个 chunk，也无法单独
解决 generated-history 分布偏移。下一步应优先实现真正局部 anchor layout，并加入 generated-
history/scheduled-sampling 训练；之后才有必要继续做 Stage2 SGF/DMD 的少步蒸馏。当前 124
帧最终视频、JSON 和训练 adapter 均已归档，根目录的并排视频保持可直接审阅。

## 2026-09-30 后续实验状态

上一阶段的 124 帧原始/causal raw-KV benchmark、并排 Demo、11 项测试和实验报告已经
完成。本轮继续实验前检查了机器状态：当前 GPU 1–5、7 仍有其他进程占用，GPU 6 约有
43 GiB 可用显存；没有终止其他用户进程。后续长实验将只使用明确空闲的 GPU，避免影响
同机任务。

本轮已完成短序列 cached/recompute 对照、CUDA 小模型训练和预训练 H3 末层 LoRA
训练及回载生成；本项目当前没有运行中的训练或 benchmark 进程。结果见下文。

### 后续实验结果（GPU 6）

当前 GPU 1–5、7 检查到有其他进程占用，因此没有终止或抢占这些任务；本轮只使用当时
约有 43 GiB 空闲的 GPU 6。

#### Cached 与 full-history recompute 对照

输出目录：`H3-World/outputs/2026-09-30-15/benchmark_cache_recompute_22_gpu6_latents/`。相同的 H3
权重、LoRA、prompt、首帧、seed 2、832×480 条件下，生成 22 帧、2 steps、5-frame
chunk、最多 5 个历史 chunk，并分别运行 raw-KV cached 和显式 full-history recompute：

| 路径 | denoiser forwards | clean commits | 采样时间 | raw-KV 峰值 | GPU allocated peak |
|---|---:|---:|---:|---:|---:|
| cached，CPU raw-KV | 4 | 2 | 30.56 s | 3,782.4 MiB | 39,601.9 MiB |
| full-history recompute | 4 | 0 | 25.63 s | 0 | 39,188.0 MiB |

两条路径都成功生成 22 帧、832×480、24 fps、H.264 视频。cached 额外执行 2 次 clean
commit，并将 100 个逐层 K/V 写入 CPU cache；在这个很短的实验上，CPU cache 搬运和
commit 开销使 cached 采样比 recompute 慢约 19%。这不是 GPU cache 加速结论，只量化了
当前实现的额外开销。

保存的最终 latent 直接比较结果为：平均绝对差 `0.010882`，95 分位绝对差 `0.03125`，
最大绝对差 `0.285156`。视频统计也接近：cached 灰度帧间 MAD 均值/p95 为 `2.054/7`，
recompute 为 `2.069/8`，空间边缘差为 `2.203` 与 `2.196`。默认混合后端运行的最终
latent 相对 L2 差异为 1.533%；完整对比保存在 `latent_comparison.json`。两条路径使用了
不同 attention 后端（SDPA/FlexAttention）、BF16 和不同计算形状，默认差异不能直接
归因为浮点误差，也不能视作真实 33B 严格等价的证据。后续统一 SDPA 诊断已将差异降到
0.502%，但仍需逐层定位剩余误差。小模型此前的 CUDA 检查约有 `1e-6` 最大误差，仅
适用于当时的小配置。

两种模式依次在同一进程执行，存在首轮编译、权重驻留与显存缓存状态差异；此时 GPU 6
也曾有其他用户小任务。表中时间属于这次运行的观测值，不用于声称稳定的 cache 加速比。

随后增加了诊断开关 `H3_CAUSAL_EAGER_SDPA=1`，让 full-history recompute 的 causal
mask 也使用 PyTorch CUDA SDPA，和 cached 路径采用相同 attention backend。该诊断运行
输出在 `outputs/2026-09-30-17/benchmark_cache_recompute_22_gpu6_sdpa/`：cached 采样 235.78 s，
recompute 采样 93.89 s；由于 dense SDPA mask 使整段路径显著变慢，这组时间不作性能
比较。latent 差异降为平均绝对差 `0.001938`、95 分位 `0.015625`、最大 `0.071777`、
相对 L2 `0.502%`。这说明前一组 1.533% 差异很大一部分来自 FlexAttention/SDPA 的
后端和归约差异，但仍有非零误差，尚不能宣称真实 H3 cached 与 recompute 严格等价。
该开关只用于正确性诊断，默认生产路径仍使用 FlexAttention。

#### Stage1 风格 teacher-forcing CUDA smoke

为了把训练验证从 8-step CPU 接口测试推进一步，`code/causal/train_smoke.py` 现在支持
`--device`，并把随机小 H3 的 attention head 改为 16 维，以满足 CUDA FlexAttention 的
最低 head dimension。GPU 6 上实际 patched `MiniMaxH3DiT` 配置为 2 层、hidden size 64、
4 个 head、head dim 16、LoRA rank 4、固定合成 latent batch，执行 256 次 clean-history
teacher-forcing flow loss：

```text
trainable parameters: 2,048
loss: 2.334223 -> 1.499979
relative decrease: 35.7%
parameters_changed: true
gradients: finite
wall time: 29.08 s
```

这证明最小 Stage1 风格 loss 在 CUDA 上可以持续优化，而不是只完成一次 forward/backward。
它仍然是随机小模型和固定合成数据，未加载预训练 33B 权重，不代表 H3 视频质量、AnyFlow
收敛或 Stage2 SGF/DMD 效果。

### 真实预训练 H3 末层 LoRA：最新结果

本轮新增 `H3-World/code/causal/train_pretrained_tail.py` 和 `pretrained_lora.py`。
使用原始 H3 30-step 生成的单个 22 帧片段作为 teacher，保存原始 latent 与条件在
`H3-World/outputs/2026-09-30-15/teacher_baseline30_22/`。冻结前 49 层和原有 action LoRA，只在
第 50 层 QKV 新增 rank-8 LoRA，共 215,040 个可训练参数，训练 80 步。

训练目标为第二个 chunk（2 个 latent frame），前 5 个 latent frame 为 clean history。
训练使用 2 个噪声 seed × 3 个 sigma（0.3/0.6/0.9），验证使用同片段的 2 个新噪声样本；
不是独立视频或跨场景验证。固定前 49 层的特征只提取一次，末层重放与整模型原始输出
最大误差为 0。

| 项目 | 结果 |
|---|---:|
| 平均训练 loss | 0.052538 → 0.040543（下降 22.8%） |
| 同片段新噪声验证 loss | 0.042231 → 0.036073（下降 14.6%） |
| 参数更新和梯度 | 参数确实改变，梯度有限 |
| 加载及特征提取 | 41.18 s |
| 80 步优化时间 | 11.90 s |
| 完整训练脚本墙钟 | 78.39 s |
| 仅末层优化阶段 allocated peak | 2,454.1 MiB |

2,454.1 MiB 不包含完整 33B 模型提取特征所需显存，不能说完整模型训练只需约 2.4 GiB。
详细记录为 `H3-World/outputs/2026-09-30-15/pretrained_tail_stage1/training.json`，adapter 为该目录
的 `adapter.pt`（约 843 KiB）。adapter 已分别回载到 22 帧、2/4 steps per chunk 的
真实 raw-KV 推理，输出在 `benchmark_trained_tail_22/` 与 `benchmark_trained_tail4_22/`。

新增可播放对照：`H3-World/outputs/h3world_30step_vs_trained_tail4_22.mp4`，左侧原始
30-step，右侧已训练的末层 causal LoRA 4-step，22 帧、1664×528、24 fps。主 124 帧
Demo 继续保留，主 Demo 的 causal 分支仍是未训练模型。

视觉观察中 2-step 的训练后输出仍明显模糊，4-step 的低层视频统计也不能支持质量改善
结论。当前验证了真实预训练权重上的“训练→保存→加载→生成”，尚未验证等质量加速。
新推理使用 `ABOT_VRAM_RESERVE_GIB=10`，GPU 外部占用随运行变化，权重驻留量也不同；
这些运行的显存和时间差不能归因为 LoRA 本身的性能收益。

最终统一测试为 `12 passed in 5.99s`，全部 causal Python 文件及相关 patched DiT/
pipeline 的语法检查通过。新增测试覆盖零初始化等价、仅末层 LoRA 有梯度及保存/加载
结果一致。上述 12 项为 CPU 测试，之前的 CUDA 小模型等价检查是另一次手动实验。

当前待解决的优先问题：真实 BF16 cached/recompute 在混合后端下有 1.533% 相对 L2，
统一 SDPA 诊断下仍有 0.502%，还需要逐层比较才能确认剩余差异来源；随后需要多片段、
首个及后续 chunk、独立验证片段上的 teacher-forcing，才能评估质量是否改善。仍未实现
AnyFlow 或 Stage2 SGF/DMD。

## 项目目标（实习生面试题）

题目是：**在 H3-World 中验证 SolarWM 的因果少步生成思路**。

需要阅读 H3-World 和 SolarWM 的 MiniMax-H3 实现，重点理解 SolarWM 的：

- 因果分块生成（causal chunks）；
- 滑动窗口局部注意力；
- 分层 KV Cache；
- Stage2 少步生成和其训练来源。

然后在 H3-World 中设计并实现一个最小 causal 模型训练/推理原型，验证这些思路
是否有机会改善长视频生成效率。最终交付：

- 一个并排对比视频：左侧为 H3-World 原始推理（目标配置例如 30 steps），右侧为
  借鉴 SolarWM 的 causal 原型；
- 相同输入、分辨率、帧数和硬件下的推理耗时与峰值显存；
- 视频质量和时间连续性比较；
- 简短实验报告，说明方法、实验、结果、可行性和局限；
- 可运行代码，以及最小 causal attention/KV-cache 和训练接口原型。

面试题明确不要求：

- 完整复现 SolarWM Stage2；
- 训练完整的 33B 模型；
- 下载或处理 SolarWM 官方约 14.45 TB latent-WDS 数据集；
- 把 SolarWM Stage2 LoRA 直接套到 H3-World 上。

重点考察能否读懂两个项目、准确定位 causal attention/KV Cache、将论文/仓库方法
迁移成可运行原型，并通过实验说明借鉴是否有效。建议周期为 1–2 周。

### 方法说明

- **SolarWM Stage0.5**：双向 flow matching。模型在完整视频片段上学习基础的视频、
  文本和相机条件表示，建立后续因果训练使用的 backbone/初始化。
- **SolarWM Stage1**：teacher forcing + AnyFlow。用干净历史 chunk 条件化当前 noisy
  目标 chunk，同时学习去噪和有限步 flow map，使模型先具备自回归因果 rollout 能力。
- **SolarWM Stage2**：DMD/self-gradient forcing。学生模型使用自己的自回归 rollout，
  frozen teacher 和 critic 提供分布匹配梯度，进一步把多步扩散/flow 轨迹蒸馏成少数几次
  denoiser evaluation。
- **Stage2 能减少采样步数的原因**：Stage2 学到的是接近数据分布的有限步 flow map，
  每次 evaluation 可以跨越更大的噪声区间，近似原来需要多次积分的小步更新。因果分块
  和 KV Cache 解决的是长视频历史条件的计算组织与复用；它们本身不等于 Stage2 蒸馏，
  不能仅凭 causal mask/KV cache 声称实现了 DMD 少步模型。

### H3-World 与 SolarWM 的关键差别

- H3-World 当前流程使用 FL2VA 的完整视频 latent，在每个 denoising step 对整段目标视频
  做一次双向 DiT attention，官方示例为 50 steps；LoRA 主要注入动作文本/attention
  条件，推理时历史 chunk 没有以 SolarWM 的 raw KV 形式跨 chunk 持久化。
- SolarWM H3 将目标视频切为 5 个 latent-frame 的 chunk，使用最多 6 个 chunk 的滑动
  窗口；当前 chunk 查询条件、音频和窗口内历史，前面 chunk 的每层 raw K/V 写入 cache，
  后续 rollout 复用；Stage1/Stage2 再分别学习 teacher-forced 和 SGF 少步更新。
- 因此本题的迁移重点是 H3-World attention 的因果可见性、chunk 边界、每层 K/V 生命周期
  和 rollout 调度，而不是复制 SolarWM 的完整数据和训练规模。

### H3-World 最小原型边界

原型应优先完成以下可验证层次：

1. 用固定 chunk 长度和窗口长度构造 H3-World 的 causal attention 可见性 mask；
2. 定义每层 raw K/V cache 的写入、读取、滑动淘汰和一致性检查；
3. 提供 teacher-forcing/flow-map 训练接口，能在少量 latent 或合成小 batch 上运行 smoke
   loss/backward；
4. 在相同输入上跑 causal inference，记录它与完整双向 baseline 的速度、显存和视频指标；
5. 明确区分“causal/KV-cache 原型”和“经过 SGF/DMD 训练的 Stage2 少步模型”。

## 当前目录结构

```text
GWM/
├── H3-World/                 # H3-World 源码仓库
├── SolarWM/                  # SolarWM 源码仓库
├── models -> /work/lpeng/qma/GWM_models
└── progress.md
```

当前用户为 `qma`。项目的用户可见路径是 `/home/lpeng/code/mq_PubDataset/GWM`，该路径是
`/home/lpeng/code/mq_PubDataset/GWM` 的符号链接；两者指向同一个工作树，
不是两个项目副本。后续命令统一使用 `/home/lpeng/code/mq_PubDataset/GWM`，共享模型链接仍然
指向 `/work/lpeng/qma/GWM_models`。

模型权重保存在共享模型目录中，没有复制大文件：

- `models/MiniMax-H3/`：约 135 GB，MiniMax-H3 FL2VA 权重；
- `models/H3-World/step-10000.safetensors`：约 131 MB，H3-World LoRA；
- `models/SolarWM/`：SolarWM H3 base 和 Stage2 checkpoint。

H3-World 内的路径链接：

- `H3-World/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3` → 共享 MiniMax-H3 权重；
- `H3-World/checkpoints/H3-World/step-10000.safetensors` → 共享 H3-World LoRA。

## 已完成

### H3-World

- H3-World 源码仓库已准备完成，包含 `code/abot/infer.py`、训练脚本和示例图片。
- 当前 `main` 与官方仓库同步，HEAD 为 `f0c7be2acbbde8256b2473f6e7b088f58272acae`。
- DiffSynth-Studio 已初始化为本地 Git checkout。
- 已固定到 H3-World 指定 revision：`300e3e4da76e881d5e6bd97d897810c18f6e4893`。
- 已应用 `H3-World/code/diffsynth_h3_action.patch`。
- 静态预检通过：
  - patched DiffSynth 从本地 `DiffSynth-Studio-h3-v2` 加载；
  - directed attention marker `leak_out` 存在；
  - per-latent action marker `action_text_spans_local` 存在；
  - MiniMax-H3 `FL2VA/model_index.json` 可见；
- LoRA checkpoint 可读取，共 208 个 tensor，全部为 LoRA 权重。
- `code/abot/infer.py` 已加入 DiffSynth `vram_limit`，默认保留 5 GiB 显存，
  通过 `ABOT_VRAM_RESERVE_GIB` 可调整；否则 48-GiB L40 会在文本编码器阶段 OOM。
- 2-step smoke inference 已成功：124 帧、832x480、seed 2，耗时 2:41.48，
  输出 `H3-World/outputs/2026-09-29-20/smoke_2step.mp4`，视频可正常解码。
- 官方 50-step baseline 已成功：相同输入和分辨率，耗时 12:12.52，GPU 0 峰值
  42179 MiB（监控最低空闲 3280 MiB），输出 `H3-World/outputs/2026-09-29-20/baseline_50step.mp4`。
- 已实现并通过独立 causal 原型单测（3 passed）：
  - `H3-World/code/causal/prototype.py`：chunk visibility、分层 raw-KV cache、
    Tiny teacher-forcing flow-map loss/backward；
  - `H3-World/tests/test_causal_prototype.py`：验证未来 chunk 不可见、窗口淘汰、
    chunk index 连续性、detach/no-grad commit 和有限 loss/梯度。
- H3 patched DiT 已增加可选 mask-only causal 开关：
  `--causal-chunk-size 5 --causal-window-chunks 5`。它复用 H3 的
  FlexAttention block mask，在 `[text | condition | audio | video | pad]` 布局上
  让视频 chunk 只能读取静态条件、当前 chunk 和窗口内历史 chunk；目前仍是
  full-sequence QKV，不是增量 KV 复用，也不是 Stage2 SGF/DMD。
- 2-step causal mask smoke 已成功：输出 `H3-World/outputs/2026-09-29-21/causal_mask_2step.mp4`，
  124 帧、832x480、24 fps、可正常解码；完整脚本墙钟约 2:03，帧间绝对差均值
  2.33、p95 8（与原始 2-step smoke 使用相同统计口径）。
- 统一输入/seed/GPU 的 30-step baseline 已完成：输出
  `H3-World/outputs/2026-09-29-21/baseline_30step.mp4`；30 次采样阶段约 5:59，VAE 解码随后完成。
  30-step causal mask 对照已完成：输出 H3-World/outputs/2026-09-29-21/causal_mask_30step.mp4；30 次采样阶段约 5:20，VAE 解码随后完成。
- 两次 30-step 采样均为 124 帧、832x480、24 fps、seed 2。GPU 0 的 nvidia-smi 采样峰值分别为 baseline 43,607 MiB、causal 43,583 MiB；差异仅 24 MiB，不能声称 causal mask 节省了显存。采样进度阶段约减少 39 秒（约 11%），但本次命令没有单独记录模型加载/采样/VAE 的结构化 timer，因此这是量级记录。
- 统一帧间绝对差统计已完成：baseline 30-step 均值 3.395、p95 12；causal mask 30-step 均值 5.355、p95 21。causal 视频连续性指标更差，说明未经 causal 训练时，直接屏蔽未来信息有质量代价。
- 已生成可播放左右并排 Demo：H3-World/outputs/h3world_30step_vs_causal_mask.mp4，1664x480、124 帧、24 fps、H.264；生成脚本为 H3-World/code/causal/make_side_by_side.py。
- 已补充实际 H3 DiT 训练可行性 smoke：随机初始化的 2-layer MiniMaxH3DiT 小配置、QKV LoRA rank 4、固定合成 latent、CPU 8 optimizer steps；loss 由 2.33863 降至 2.33549，1,536 个 LoRA 参数更新且梯度有限。该 smoke 只验证 forward/backward 接口和冻结逻辑，不代表预训练 H3 视频质量。
- causal 原型测试已扩展为 12 个：包含 mask oracle、窗口淘汰、detach/no-grad、cache 与显式 clean history 一致性、实际 H3 DiT 的未来帧隔离和 LoRA 梯度测试、cache eviction、末尾不足完整 chunk、CPU full-history recompute、adapter 梯度及保存加载；pytest -q 四组 causal 测试已通过（12 passed）。
- 已写实验报告：docs/causal_prototype_report.md，包含 Stage0.5/Stage1/Stage2 解释、H3/SolarWM 差异、实现边界、统一视频结果和局限性。
- 已阅读并定位 SolarWM MiniMax-H3 的因果实现：`H3RawKVCache`、5-frame chunk、
  最多 5 个历史 chunk 的滑动窗口、rollout/replay attention，以及 Stage1
  teacher-forcing/AnyFlow 和 Stage2 SGF/DMD 的调用边界。
- 已阅读并定位 H3-World patched DiT 的 `MiniMaxH3Attention`、FlexAttention
  `block_masks`、`MiniMaxH3DiT.forward` 和 `model_fn_minimax_h3`。当前 H3-World
  的默认路径仍是整段视频在每个 denoising step 做 full-sequence attention；没有
  自动复用跨 chunk 的 raw K/V。

### 真实 chunk + raw-KV 实验（已完成）

在上述 mask-only 原型之后，已经完成真正按 chunk rollout、逐层 raw K/V 提交和窗口
淘汰的 124 帧 H3 实验。公共条件保持一致：`examples/first_frame.png`、同一 prompt、
`forward` action、seed 2、832×480、24 fps、124 帧；三次运行分别使用空闲的 NVIDIA
L40 GPU 3、GPU 1、GPU 2，结果保存在 `H3-World/outputs/`：

| 配置 | 输出 | denoiser forwards | 采样时间 | VAE prepare/decode | 峰值显存 | raw-KV 峰值 | 总墙钟（conditioning 后） |
|---|---|---:|---:|---:|---:|---:|---:|
| 原始 H3，30 steps | `benchmark_baseline_124/baseline.mp4` | 30 | 374.57 s | 15.87 / 11.66 s | 34,958 MiB allocated；37,820 MiB reserved | 0 | 424.09 s |
| 原始 H3，4 steps | `benchmark_baseline4_124/baseline.mp4` | 4 | 55.10 s | 25.24 / 10.37 s | 37,748 MiB allocated；40,452 MiB reserved | 0 | 116.11 s |
| causal chunk + raw-KV，4 steps/chunk | `benchmark_cached_cpu_124/cached.mp4` | 32；另有 8 次 clean commit | 231.69 s | 19.32 / 11.90 s | 37,748 MiB allocated；38,782 MiB reserved | 13,508.6 MiB（CPU） | 284.89 s |

cached 配置为 5 latent-frame chunk、最多 5 个历史 chunk、8 个 chunk、固定 audio noise、
全局 H3 RoPE。每个 chunk 的 4 次采样完成后，额外以 `sigma=0` 做一次 clean forward
写入 cache；不能把最后一个 noisy step 的 K/V 当作历史。8 个 chunk × 50 个 Transformer
层得到 400 次 per-layer commit。cached 视频和两个 baseline 均为 124 帧、5.1667 秒、
H.264，可正常解码。

这次实验证明 33B H3 的 causal chunk 调度和 raw-KV 生命周期在工程上可以跑通，但当前
CPU cache 不是性能优化：它比原始 4-step 全序列路径多出 chunk 级 forward 和 cache 搬运，
总墙钟为 284.89 s；与 30-step 原始路径相比采样阶段减少到 231.69 s，但这不能归因于
Stage2，也不能作为同等质量的加速结论。cached 视频存在背景模糊、场景变形和人物细节
丢失，原因是模型没有经过 Stage1 causal teacher-forcing 或 Stage2 SGF/DMD 训练。

曾在 GPU 2 尝试把 raw-KV 放到 CUDA：第一个 chunk 后 cache 约占 2.64 GiB，下一层
权重加载时仅剩约 443 MiB，申请约 498 MiB 失败并 OOM。因此单张 48-GiB L40 在当前
33B 逐层加载策略下不能同时容纳完整 GPU cache；后续需要 CPU offload、KV 压缩、权重/
cache 分层或多 GPU 才能评估 GPU cache 的真实收益。当前不能宣称已经实现 GPU KV-cache
加速。

质量/连续性统计写入 `H3-World/outputs/2026-09-30-14/benchmark_video_metrics.json`：原始 30-step
的灰度帧间 MAD 均值/p95 为 3.322/12，原始 4-step 为 2.666/9，causal raw-KV 4-step
为 2.986/11；空间边缘差分别为 2.166、2.114、2.939。MAD 较低不等于质量更好，必须
结合视频观察；当前 cached 输出的视觉退化是明确的失败边界。

最终面试题 Demo 已生成：
`H3-World/outputs/h3world_30step_vs_cached4step.mp4`，左侧是原始 H3 30 steps，右侧
是 causal chunk + raw-KV prototype 4 steps/chunk，1664×528、124 帧、24 fps、H.264，
带顶部标签栏。生成脚本仍为 `H3-World/code/causal/make_side_by_side.py`。

原型测试目前共 12 项，新增覆盖 cache eviction、末尾不足完整 chunk、adapter
保存加载和末层梯度约束，以及 CPU
full-history recompute 与 CUDA cached attention 的数值一致性；最近一次完整结果为
`12 passed`（包含 adapter 测试）。

三份 benchmark 视频均为 124 帧、24 fps、832x480、约 5.17 秒；baseline 和 causal
raw-KV 的帧间差、空间边缘差均已使用相同脚本统计。

本项目不下载 SolarWM 官方完整 latent-WDS。该发布包约 14.45 TB，面向完整
SolarWM Stage0.5/Stage1/Stage2 训练和官方数据集评测；当前目标是把因果 chunk、
局部注意力和 KV-cache 思路移植到 H3-World 的最小原型，不要求完整复现 Stage2。
因此不需要 ModelScope 登录、TB 级 latent 数据或官方训练数据集。

### Python 环境

已创建两个隔离环境，避免 H3-World 与 SolarWM 的 CUDA/PyTorch 依赖冲突。
环境由当前用户 `qma` 创建，均使用 uv 管理的 CPython 3.10.21，实际路径为
`/home/lpeng/code/mq_PubDataset/GWM/.venvs/`；系统 `/usr/bin/python` 的 Python 3.12 不用于这两个环境。
两个 venv 中都额外安装了 `pip==26.2.1`，既可以使用 uv，也可以在激活后使用
`python -m pip`。

#### `h3world`

- 路径：`/home/lpeng/code/mq_PubDataset/GWM/.venvs/h3world`
- Python 3.10.21
- PyTorch 2.10.0+cu128
- torchvision 0.25.0+cu128
- torchaudio 2.10.0+cu128
- H3-World `requirements.txt` 已安装，包括 Diffusers 0.37.1、Transformers 4.57.3、Accelerate 1.14.0、PEFT 0.20.0 等。
- PyTorch CUDA 可用，检测到 8 张 NVIDIA L40。
- patched DiffSynth 静态 preflight 通过，实际从 `H3-World/DiffSynth-Studio-h3-v2` 加载。

#### `solarwm-h3`

- 路径：`/home/lpeng/code/mq_PubDataset/GWM/.venvs/solarwm-h3`
- Python 3.10.21
- PyTorch 2.6.0+cu124
- Diffusers 0.40.0
- Transformers 5.12.1
- PEFT 0.20.0
- FlashAttention 2.8.3 已安装
- SolarWM 已 editable 安装。
- `solarwm environment probe` 已通过，CUDA/NCCL/核心包版本可识别。
- 当前 `main` 与官方仓库同步，HEAD 为 `ce1da4e7705391eda8eeda6016c0fd3f614b975e`。

## 已知问题

SolarWM 环境中的 FlashAttention 2.8.3 扩展目前无法在 PyTorch 2.6.0+cu124 下直接导入，报 C++ ABI 错误：

```text
undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_string...
```

SolarWM 的 environment probe 已检测到该问题并回退到 PyTorch native attention，因此核心适配器可以导入；本机没有 `nvcc`，无法直接从源码重编 CUDA 扩展。正式 Stage2 性能实验前，需要在有匹配 CUDA toolkit 的环境中修复，或明确记录 native attention 基线，否则不能声称使用了 FlashAttention 加速。

SolarWM 的 `decord==0.6.0` 可以正常 import，但其 PyPI wheel 的旧平台标签会使
`python -m pip check` 报告 `built for a different platform`；这是 wheel 元数据问题，
不影响当前 import。由于 PyPI 没有对应的 source distribution，暂不在本机重编 decord。

## 尚未完成

- 尚未完成 Stage1 AnyFlow/完整数据规模的 teacher-forcing 训练和 Stage2 SGF/DMD 少步
  蒸馏；当前 raw-KV 视频仍不能称为 Stage2 模型。已完成随机小模型 CUDA smoke，以及
  单片段预训练 H3 末层 QKV LoRA 的有限可行性实验。
- 尚未实现可在单张 48-GiB L40 上运行的 GPU raw-KV 版本；CPU cache 已完成完整长视频，
  GPU cache 的 OOM 边界已记录。
- 尚未证明 causal 少步模型在相同视频质量下优于原始 H3；当前 cached 视频的视觉质量
  明显下降，音频输出也固定为静音实验路径。

## 下一步计划

1. 在少量 H3 latent 上做 Stage1 风格 clean-history teacher forcing/AnyFlow，再比较
   4-step/8-step student 与 30-step baseline；不要把当前 raw-KV 结果称为 Stage2。
2. 优化 cache CPU offload、KV 压缩、权重/cache 分层或多 GPU，重新测量 GPU cache 的
   实际收益和显存占用。
3. 若需要 SolarWM runtime 对照，再处理 FlashAttention ABI；当前 native attention
   限制必须在报告中保留，且不是 H3 causal 原型的必需条件。

### 预训练 adapter 的 4-step 回载验证

同一 `outputs/2026-09-30-15/pretrained_tail_stage1/adapter.pt` 又在 4 steps/chunk、22 帧上完成回载
验证，输出为 `H3-World/outputs/2026-09-30-15/benchmark_trained_tail4_22/cached.mp4`：8 次 denoiser
forward、2 次 clean commit，采样 78.99 s，conditioning 后总墙钟 122.24 s，GPU
allocated peak 25,895.4 MiB，CPU cache 峰值 3,782.4 MiB。它证明 adapter 在多步
cached rollout 中可以稳定加载；这仍然不是 Stage2 少步蒸馏，也没有质量优越性结论。

## 运行环境注意事项

- 当前用户可见工作目录：`/home/lpeng/code/mq_PubDataset/GWM`，物理路径为 `/home/lpeng/code/mq_PubDataset/GWM`；二者是同一工作树。
- 两个 Python 环境：`/home/lpeng/code/mq_PubDataset/GWM/.venvs/h3world` 和 `/home/lpeng/code/mq_PubDataset/GWM/.venvs/solarwm-h3`。
- 不要把代码路径写成 `/work/lpeng/qma/GWM`；该位置只用于共享模型存储的真实目标路径。
- H3-World 必须加载固定 revision 加 patch 的 DiffSynth，不能使用未修改的普通 DiffSynth。
- H3-World 不要 editable install DiffSynth；脚本会检查实际加载的 patched checkout。
- 已完成 2-step/30-step H3 推理、CPU/GPU 小模型训练 smoke、单片段预训练 H3 末层
  LoRA smoke、124 帧 raw-KV benchmark 和 12 项测试；不要在无明确需要时重新启动长时间生成任务。

### 2026-09-30 本轮新增实验

12 项测试仍为通过状态；`code/causal/*.py` 已通过 `py_compile`。为避免损坏的 teacher
被训练脚本静默使用，`code/causal/benchmark.py` 新增 finite 检查：conditioning 完成后
会检查 prompt embedding、video/audio latent 和首帧 anchor，任一 NaN/Inf 都立即失败。

在 seed 2 的 22 帧 teacher clip 上新增 `target_chunk=0` 训练：冻结前 49 层和原有
action LoRA，只训练第 50 层 rank-8 QKV LoRA，共 215,040 个参数、80 步。frozen-feature
replay 最大误差为 0；train loss `0.102748 → 0.082926`（下降 19.3%），validation
`0.083897 → 0.071582`（下降 14.7%）；特征提取 74.17 s，优化 10.83 s，优化阶段
allocated 峰值 2,213.9 MiB。adapter 为
`H3-World/outputs/2026-09-30-22/pretrained_tail_chunk0/adapter.pt`。结合此前 seed 2 的 chunk 1
结果（train 下降 22.8%、validation 下降 14.6%），说明接口同时覆盖无历史首 chunk 和
有 clean 历史的后续 chunk，但仍只是单片段、有限噪声样本。

为做最小独立随机复核，在空闲 GPU 1 上生成了 seed 11 的 22 帧、30-step teacher：
`H3-World/outputs/2026-09-30-22/teacher_baseline30_22_seed11/`。conditioning/latent 均 finite，
采样 146.72 s，allocated 峰值 34,701.8 MiB。对该 clip 的 chunk 1 训练结果为：train
`0.058336 → 0.044018`（下降 24.5%），validation `0.045028 → 0.038609`（下降
14.3%），特征提取 53.53 s，优化 6.64 s，优化阶段 allocated 峰值 2,454.6 MiB，
replay 最大误差 0；adapter 为
`H3-World/outputs/2026-09-30-22/pretrained_tail_seed11_chunk1/adapter.pt`。这只是同一场景不同随机
teacher 的有限复核，尚未构成独立场景泛化或质量提升证据。

一次 GPU 3、seed 7 的 teacher 运行出现 prompt/video NaN，mp4 仅几 KB，已判为无效并
未用于训练；该失败促成上面的 finite 保护。当前没有本项目后台进程，外部 GPU 任务未被
终止。

随后把 seed 11 adapter 回载到 4-step/chunk causal rollout，输出
`H3-World/outputs/2026-09-30-22/benchmark_trained_seed11_tail4_22/cached.mp4`，并生成并排 Demo
`H3-World/outputs/h3world_30step_vs_trained_seed11_tail4_22.mp4`。该 rollout 为 22 帧、
8 次 denoiser forward、2 次 clean commit，采样 47.03 s，conditioning 后总墙钟 89.53 s，
GPU allocated 峰值 34,702.6 MiB，CPU raw-KV 峰值 3,782.4 MiB，100 个逐层 commit。它
证明独立随机 teacher 的 adapter 可加载到多步 causal 调度；仍没有质量等价或 Stage2
少步蒸馏结论。

seed 11 teacher 与 causal 输出的描述性统计也已写入
`H3-World/outputs/2026-09-30-23/seed11_video_metrics.json`：teacher 帧间灰度 MAD 均值/p95 为
`3.578/13`、空间边缘差 `2.233`；causal 为 `3.149/11`、`2.045`。这些指标不能替代
VMAF、光流或人工评价，低 MAD 也可能来自模糊或运动不足。

### 下一阶段路线（2026-09-30）

下一步先做 243 帧（约 10.1 秒）的 paired diagnostic：同一 prompt、首帧、seed 下分别
运行原始 H3 30-step 和 causal raw-KV 4-step，并记录 chunk 边界附近的帧间 MAD、长期
漂移、denoiser 次数、CPU KV 峰值和 GPU 峰值。243 帧对应 72 个 latent frame、15 个
5-frame chunk；4-step causal 预计为 60 次 noisy denoiser forward 加 15 次 clean commit。
先用 243 帧而不是 481 帧，是为了在成本可控的情况下确认 5 秒视频中观察到的跳变是否
在更长时间上累积。

243 帧实验完成后，进入真正的 Stage1 风格多 chunk teacher-forcing：训练数据至少覆盖
首 chunk、带 clean history 的后续 chunk 和一个 held-out clip；比较 4-step/8-step
causal rollout 与 30-step teacher 的边界连续性和画质。只有 student 能在少步下稳定跟随
teacher 后，才考虑 Stage2 风格的轨迹蒸馏；当前 mask/cache inference 仍不能称为 Stage2。

本次资源检查发现 GPU 1、2、4 有外部任务，GPU 3 只有约 28 GiB 空闲，未启动长任务，
也未终止任何外部进程。等到有至少约 40 GiB 可用显存的空闲 L40 后再运行 243 帧配对
benchmark。

### 2026-10-01 00 时段：多 chunk Stage1 风格实验

用户提供了三张空闲 L40 后，新增脚本
`H3-World/code/causal/train_pretrained_multichunk.py`。它在真实预训练 H3 上冻结前
49 层和已有 action LoRA，只训练第 50 层 rank-8 QKV LoRA；同一个 adapter 同时使用
chunk 0（无历史）和 chunk 1（前 5 个 latent frame 为 clean history），每个 chunk 有
6 个训练噪声样本和 2 个验证噪声样本，共 120 步。该实验严格标记为 Stage1-style
clean-history flow matching，不包含 AnyFlow、Stage2 SGF/DMD 或少步 student 蒸馏。

本时段所有原始产物统一放在
`H3-World/outputs/2026-10-01-00/`，目录说明见其中的 `README.md`。三张卡并行完成：

| teacher | train loss | validation loss | replay 最大误差 | adapter |
|---|---:|---:|---:|---|
| seed 2 | `0.077643 → 0.062125`（-20.0%） | `0.063064 → 0.053967`（-14.4%） | 0 | `pretrained_multichunk_seed2/adapter.pt` |
| seed 11 | `0.083957 → 0.066981`（-20.2%） | `0.067879 → 0.058830`（-13.3%） | 0 | `pretrained_multichunk_seed11/adapter.pt` |
| seed 13（本时段新生成） | `0.077194 → 0.061457`（-20.4%） | `0.063288 → 0.054377`（-14.1%） | 0 | `pretrained_multichunk_seed13/adapter.pt` |

三次训练均参数发生改变、梯度有限、没有 OOM；每次可训练参数 215,040，优化阶段
allocated 峰值约 2,453 MiB，完整脚本墙钟约 98–109 s（不含同一进程的完整模型驻留
解释）。seed 13 adapter 已回载到真实 cached rollout，原始结果位于
`2026-10-01-00/benchmark_multichunk_seed13_tail4_22/`：22 帧、4 steps/chunk、8 次
denoiser forward、2 次 clean commit，采样 37.98 s，conditioning 后总墙钟 77.29 s，
GPU allocated 峰值 34,702.6 MiB，CPU raw-KV 峰值 3,782.4 MiB。

本时段最终需要查看的对比视频单独放在 `H3-World/outputs/` 根目录：
`h3world_30step_vs_multichunk_seed13_tail4_22.mp4`。其他最终 Demo（例如
`h3world_30step_vs_cached4step.mp4`）也继续只放在根目录；训练、teacher、benchmark
原始数据不再直接堆在根目录。

需要明确标记：seed 13 multi-chunk 视频不是 Stage1 成功效果。该视频只有 22 帧（约
0.92 秒），右侧 causal 输出肉眼几乎停在原地；teacher 的灰度帧间 MAD 均值/p95 为
`3.490/12`，causal 为 `2.905/11`，与运动被压平的观察一致。三次多 chunk 训练的
loss 下降、replay=0 只证明冻结特征上的单次 clean-history flow prediction 可以优化，
没有证明 free-running rollout 能保持运动。

失败原因已写入实验报告：当前每个 clip 只有 12 个训练噪声和 4 个验证噪声，只训练最后
一个 DiT block 的 QKV LoRA；训练用固定 clean history 和 `sigma={0.3,0.6,0.9}`，推理
用生成历史和 4-step scheduler；训练没有对多步自由运行、动作保持或 chunk 边界施加约束。
因此这不是 SolarWM Stage1 的完整效果，而是 Stage1-style 接口的负结果。下一轮应先统一
8/16-step 的训练和推理时间表、扩大多 clip 数据、训练所有 causal QKV block，并先以
8-step free-running rollout 作为验收，再考虑 4-step。

为隔离采样步数影响，同一个 seed 13 multi-chunk adapter 又做了 8 steps/chunk 回载：
原始数据在 `2026-10-01-00/benchmark_multichunk_seed13_tail8_22/`，采样 62.02 s，
16 次 denoiser forward、2 次 clean commit，GPU allocated 峰值 34,702.6 MiB，CPU
raw-KV 峰值 3,782.4 MiB。8-step 输出的帧间灰度 MAD 均值/p95 为 `2.422/11`，低于
4-step 的 `2.905/11`，抽帧仍近似静止；增加采样步数没有恢复运动，说明主因是训练目标
与 free-running causal rollout 的分布不匹配，而非单纯 4 步不足。8-step 诊断视频
`h3world_30step_vs_multichunk_seed13_tail8_22_diagnostic.mp4` 已放在 outputs 根目录，
同样标记为负结果。

随后完成了整个 `H3-World/outputs/` 的归档整理：历史原始目录和单独的 teacher/benchmark
文件也按香港本地修改时间移动到 `YYYY-MM-DD-HH/` 二级目录（例如
`2026-09-30-15/`、`2026-09-30-22/`），根目录现在只保留 `README.md` 和
`h3world_*.mp4` 最终并排对比视频。每个时间目录都有自己的 `README.md`，根目录索引为
`H3-World/outputs/README.md`；报告中的历史路径已同步更新。

### 2026-10-01 05–06 时段：Stage1 双 anchor 原型和 tail4 容量复测

本时段继续围绕面试题的 Stage1 主线，加入了用户提出的“上一 chunk 最后一帧作为图片条件，同时保留 action”的实现。单独替换原始首帧会丢失全局场景锚点，因此最终采用双 anchor 协议：slot 0 永远保留 H3 原始首帧 image anchor，slot 1 在第 0 个 chunk 先放首帧副本，后续 chunk 替换成上一 chunk 的最后一个 latent frame；slot 1 的 H3 temporal RoPE position 会被 retime 到真实的全局前置帧位置。action text 仍只绑定当前 latent frame，历史 action 不会被当前 video query 读取。

代码改动：

- `H3-World/code/causal/h3_cached.py` 增加并完善 `expand_packed_two_anchors()`，将旧的单 image-anchor packed layout 扩展为双 slot，并保持 audio/text/video index、padding 和 action metadata 的一致性；chunk 0 的副本现在与首帧使用同一个时间位置，后续调用 `retime_anchor_position(..., anchor_slot=1)`。
- `H3-World/code/causal/benchmark.py` 新增 `--anchor-mode dynamic_last_frame_dual`，cached/recompute rollout 和 clean KV commit 使用同一套双 anchor 规则；baseline 被明确禁止使用该 causal-only 模式。
- `H3-World/code/causal/train_pretrained_multichunk.py` 同步支持双 anchor，训练时使用 clean previous-history tail，推理时使用同样的第二 slot，避免训练和 rollout 的 packed layout 不一致。
- `H3-World/tests/test_h3_cached.py` 新增双 anchor index remap、原始 slot 保持不变和 slot 1 全局 retime 测试。

验证结果：`py_compile` 通过，完整测试为 **14 passed**；GPU1 上的双 anchor 22 帧 smoke rollout 也成功完成（1 step/chunk、2 次 clean commit、100 个逐层 cache commit），结果在 `H3-World/outputs/2026-10-01-08/dual_smoke_22_corrected/`。

完成了两个 124 帧（5.17 秒）tail4 Stage1-style 复测，均为 8 steps/chunk、scheduler shift 2.22、rank-8 QKV LoRA、160 optimizer steps、chunks 0–7：

| 配置 | train loss | validation loss | rollout 灰度 MAD 均值/p95 | 空间边缘差 | adapter |
|---|---:|---:|---:|---:|---|
| 4 blocks，dynamic last frame | `0.10522 → 0.08495` | `0.10618 → 0.09324` | `3.786 / 15` | `2.013` | `outputs/2026-10-01-06/stage1_dynamic_shift2_tail4/adapter.pt` |
| 双 anchor，4 blocks，dynamic last frame | `0.09880 → 0.07969` | `0.10231 → 0.09009` | `4.225 / 16` | `2.074` | `outputs/2026-10-01-08/stage1_dual_shift2_tail4/adapter.pt` |
| H3 原始 30-step teacher | — | — | `4.295 / 15` | `2.226` | `outputs/2026-10-01-01/baseline30_seed13_124/baseline.mp4` |

完整 dual-anchor rollout 位于 `H3-World/outputs/2026-10-01-08/stage1_dual_shift2_tail4_rollout_124/cached.mp4`，三者统计位于 `H3-World/outputs/2026-10-01-08/metrics_124_shift2_dual_tail4.json`；单 adapter 的统计位于 `H3-World/outputs/2026-10-01-06/metrics_124_shift2_tail4.json`。根目录的最终并排视频尚未替换，避免把仍有明显长程漂移的 dual 结果误标成最终成功结果。

当前结论：双 anchor 的架构是合理的，且 clean-history Stage1 训练链路、cache、位置重映射已经真实跑通；但在单场景、少量样本、仅 4 个 tail blocks 的条件下，free-running rollout 仍在约第 30 帧以后出现重影、亮度漂移和 chunk 边界突变，dual 版本没有证据表明它已经优于单 anchor。tail4 单 anchor 的整体 MAD 较低但在第 34、85、102 帧附近有较大边界峰值，dual 版本保留了更多运动幅度却在长程上仍然模糊。这说明瓶颈仍是 clean-history teacher forcing 与 generated-history rollout 的分布差异，以及当前 raw latent patchify 并非真正的 RGB decode/re-encode image anchor；不能把该结果称为 SolarWM Stage2，也不能声称已经实现少步质量等价。

下一步应保持这套双 anchor 作为可复现实验分支，优先做 scheduled/noisy-history 或多 clip 训练，并对 chunk boundary 加入显式 continuity loss；如果继续使用双 anchor，应再比较真正的 RGB decode/re-encode anchor，而不是修改随机种子或把 ordinary 8-step solver 误称为 Stage2。

### 2026-10-01 06–07 时段：scheduled/noisy-history 诊断

为直接检查 generated-history 分布偏移，使用相同 124 帧 teacher、双 anchor、4 个 tail blocks、shift 2.22 和 160 步训练，给 clean history 加 `history_noise_std=0.08`，结果在 `H3-World/outputs/2026-10-01-09/stage1_dual_shift2_tail4_historynoise08/`。训练本身成功，train loss `0.10213 → 0.08516`、validation `0.10841 → 0.09908`、replay error `0`；但 8-step free-running 回载的 MAD/边缘差为 `5.362/21`、`2.773`，明显差于 clean-history dual adapter 的 `4.225/16`、`2.074`，并且第 102、119 帧边界峰值更大。

这次结果不支持“简单给历史 latent 加高斯噪声就能解决 rollout drift”。它仍然是有效的 Stage1 负诊断：scheduled history 需要模拟真实多步 student/generated history 的结构性误差，而不是只增加独立像素噪声。当前保留 clean-history dual adapter 作为主架构示例，noisy-history adapter 作为失败对照；下一轮应考虑 chunk-level rollout unroll、边界 continuity loss 和真实 RGB image anchor。

### 2026-10-01 07 时段：persistent cached-history Stage1 修正与双 anchor 长片复测

上一轮尝试把训练端改为持久化 causal KV cache 时，在回放训练特征的阶段发现了一个实现错误：
尾部 4 个 DiT block 都复用了第一个尾 block 的 `layer_index`，使第 47–49 层读取了第 46
层的历史 K/V。该轮输出位于 `outputs/2026-10-01-11/`，已标记为 `INVALID.md`，adapter
和任何 rollout 都不作为实验结论。

修正内容：`causal/pretrained_lora.py` 新增 `replay_tail()`，按实际 block index 逐层回放；
训练脚本恢复严格的 zero-initialized adapter replay 检查。小模型回归和完整测试均通过，训练
端的 replay 最大误差重新为 `0`。这一步很关键：Stage1 的历史 KV 必须按层保存和读取，不能
把“cache 能运行”误认为“cache 语义正确”。

修正后的真实 H3 训练产物在
`outputs/2026-10-01-12/stage1_causal_cache_dual_tail4/`：124 帧 teacher、chunk 0–7、
4 个末端 DiT blocks、rank-8 QKV LoRA、160 optimizer steps、H3 scheduler `shift=2.22`、
1 个训练噪声 seed 和 2 个验证 seed。结果为：

| 指标 | 结果 |
|---|---:|
| train loss | `0.10318 → 0.08088` |
| validation loss | `0.10897 → 0.09149` |
| train/validation samples | `32 / 16` |
| replay max error | `0` |
| trainable parameters | `860,160` |
| optimizer wall time | `84.4 s` |

回载 rollout 位于
`outputs/2026-10-01-12/stage1_causal_cache_dual_tail4_rollout_124/cached.mp4`。它使用
8 steps/chunk、8 个 5-frame chunk，执行 `64` 次 noisy denoiser 和 `8` 次 clean KV commit；
采样时间 `323.55 s`，GPU allocated 峰值约 `39.6 GiB`，CPU raw-KV 峰值约 `13.2 GiB`。
与同 seed 的原始 H3 30-step teacher 对比：

| 输出 | 灰度帧间 MAD 均值/p95 | 空间边缘差 |
|---|---:|---:|
| H3 30-step teacher | `4.295 / 15` | `2.226` |
| corrected cached-history Stage1-style | `4.277 / 16` | `2.275` |

整体运动幅度已经接近 teacher，但 chunk 边界相邻帧仍有明显峰值（例如 decoded frame 34、
68、85、102 附近），所以这仍是 Stage1-style feasibility result，不是质量等价结果，也
不是 SolarWM Stage2。根目录可直接播放的并排视频是
`outputs/h3world_30step_vs_stage1_cached_history_dual_tail4_8step_124.mp4`。

### 上一末帧作为图片条件：latent 与真实 RGB 两条路径

用户提出的“每个新 chunk 额外给上一 chunk 最后一帧，并继续给 action”已经落成双 anchor
协议：slot 0 保留 H3 原始首帧，slot 1 在 chunk 0 用首帧副本，后续 chunk 使用上一段可用
历史的末帧；action text 仍按当前 latent frame 绑定，因此不会因为增加图片 anchor 而丢掉
动作条件。现有 Stage1 adapter 训练的是 normalized temporal-latent patchify 版本。

另外新增了 `last_frame_image_anchor()` 和 benchmark 的
`--anchor-mode dynamic_last_frame_rgb_dual`。它严格走 H3 图片条件语义：把当前已经生成的
latent prefix 解码为 RGB，取 prefix 的最后一个可见 RGB frame，再用
`encode_video(process_image=True)` 重编码并 patchify；不会把 temporal latent 直接冒充图片
latent。由于 H3 temporal VAE 需要上下文，不能只 decode 一个 latent token 后取第一张补帧。

22 帧、4 steps/chunk 的 RGB-prefix smoke 位于
`outputs/2026-10-01-12/rgb_prefix_smoke_22/`，VAE anchor 额外耗时约 `25.4 s`。相对于
同一 22 帧 30-step teacher，它的描述性统计是 MAD `2.573/8`、空间边缘差 `2.754`；这只是
未训练的 causal anchor ablation，较低 MAD 不能解释为更高质量。它证明了“上一末帧 + action”
可以用 H3 原生 image branch 实现，下一步若要验证质量，必须把同样的 RGB-prefix 协议纳入
Stage1 teacher forcing，而不能训练 latent anchor 后再只在推理时替换编码方式。

本时段最后的代码检查为：`pytest -q H3-World/tests` **16 passed**，
`python -m py_compile H3-World/code/causal/*.py` 通过。当前主线仍是：先完成严格对齐的
Stage1 causal training 和长视频连续性，再考虑 Stage2 的 SGF/DMD trajectory distillation；
普通 8-step solver、causal mask 或 KV cache 本身都不应被称为 Stage2。

### 2026-10-01 08–09 时段：RGB prefix anchor 的正式 Stage1 训练与长片对比

为真正验证“上一 chunk 最后一帧作为图片条件”的想法，而不是只在推理时替换条件，训练脚本
新增 `--anchor-mode dynamic_last_frame_rgb_dual`。对于每个 clean-history chunk，训练端先
解码当前可用 latent prefix，取 prefix 的最后 RGB frame，再用 H3 的
`encode_video(process_image=True)` 重新编码 slot 1；rollout 端使用完全相同的 prefix 规则。
slot 0 仍保留原始 H3 首帧，action text 仍只绑定当前 latent frame。

RGB-prefix Stage1 训练产物在
`outputs/2026-10-01-13/stage1_causal_cache_rgb_dual_tail4/`：

| 指标 | latent dual anchor | RGB prefix dual anchor |
|---|---:|---:|
| train loss | `0.10318 → 0.08088` | `0.09810 → 0.08012` |
| validation loss | `0.10897 → 0.09149` | `0.10255 → 0.09040` |
| replay max error | `0` | `0` |
| trainable parameters | `860,160` | `860,160` |

RGB 124 帧 free-running rollout 位于
`outputs/2026-10-01-13/stage1_causal_cache_rgb_dual_tail4_rollout_124/cached.mp4`，
对应并排视频为根目录的
`h3world_30step_vs_stage1_rgb_cached_history_dual_tail4_8step_124.mp4`。它执行同样的
64 次 noisy denoiser 和 8 次 clean commit，但由于每个后续 chunk 要切换 VAE 并编码图片，
采样时间增加到 `674.72 s`；7 次 RGB anchor 的额外 VAE 处理合计约 `202.8 s`，CPU raw-KV
峰值仍约 `13.2 GiB`。

与原始 30-step teacher 的统计：

| 输出 | 灰度帧间 MAD 均值/p95 | 空间边缘差 |
|---|---:|---:|
| H3 30-step teacher | `4.295 / 15` | `2.226` |
| latent dual cached-history | `4.277 / 16` | `2.275` |
| RGB-prefix dual cached-history | `4.200 / 16` | `2.138` |

RGB 版本整体运动幅度和边缘差略低于 teacher，且没有证据显示它解决长期边界漂移；后段边界
相邻帧 MAD 仍可达到 `13–17`（frame 85、102 附近）。因此当前实验回答是：**上一末帧作为
图片条件在接口和训练语义上可行，但在单场景、小数据、只训练 4 个尾 blocks 的 Stage1
原型中没有带来可确认的质量或连续性提升；它还显著增加了推理 VAE 开销。** 这不是 RGB
方案无效的最终结论，因为还没有做 generated-history unroll、多 clip 数据或 continuity loss。

本时段 RGB 训练和 rollout 均完成，所有新增 smoke/long-run 文件按小时归档；根目录保留两条
可播放长片，便于直接比较 latent dual 与 RGB dual 的差异。

## 2026-10-01 08–09 时段：mixed generated-history Stage1 诊断与当前主 Demo

本时段继续完善 Stage1 训练协议，直接检查“clean-history teacher forcing 与 free-running
generated-history 不一致”是否可以通过按比例混合历史来缓解。`train_pretrained_multichunk.py`
新增：

```text
--history-latents PATH
--history-mix 0..1
```

其中 `history-mix=0` 是 teacher clean history，`history-mix=1` 是给定 detached causal
rollout history；当前 chunk 的 supervised flow target 仍然来自原始 30-step teacher，没使用
未来 chunk，也没有 SGF/DMD 或 Stage2 student。这个开关只用于 scheduled-history 诊断。

历史来源使用 2026-10-01 16 时段的双 anchor、blend=0.35、overlap=2 rollout latent。两套
训练都覆盖 124 帧的 chunk 0–7，末 4 个 DiT block 使用 rank-8 QKV LoRA，H3 scheduler
shift=2.22，160 optimizer steps：

| 目录 | history mix | train loss | validation loss | replay max error |
|---|---:|---:|---:|---:|
| `outputs/2026-10-01-17/stage1_mixed_history025_dual_tail4/` | 0.25 | `0.11228 → 0.08230` | `0.11542 → 0.09854` | `0` |
| `outputs/2026-10-01-17/stage1_mixed_history050_dual_tail4/` | 0.50 | `0.14743 → 0.10088` | `0.13406 → 0.11759` | `0` |

两个 adapter 均通过逐层 zero-initialized replay 检查，并回载到同 seed 13、124 帧、8
steps/chunk 的真实 cached rollout。结果如下：

| 输出 | 灰度 MAD 均值/p95 | 空间边缘差 | sampling |
|---|---:|---:|---:|
| H3 30-step teacher | `4.295 / 15` | `2.226` | 原始 teacher |
| mixed history 0.25 | `5.207 / 20` | `4.100` | `315.84 s` |
| mixed history 0.50 | `6.479 / 24` | `4.318` | `278.99 s` |

抽帧接触图显示两套 mixed-history 分支运动比之前近似静止的负结果更明显，但停车场结构
在后续 chunk 更快偏移，不能作为质量改善。结论是：把一条已有 generated rollout 以固定
比例混入单场景 teacher history 没有解决分布偏移，反而放大了 drift；这两套结果保留为
Stage1 负对照。它们都不是 Stage2，因为没有 trajectory distillation、AnyFlow 或 SGF/DMD。

### 边界诊断（已由后续 tail16 主 Demo 超越）

该边界诊断视频仍保留用于分析 soft overlap，但当前根目录主 Demo 已在后续 09–10 时段
切换为 tail16 causal 适配版本：

```text
H3-World/outputs/h3world_30step_vs_stage1_cached_history_dual_anchor_softoverlap2_8step_124.mp4
```

它对应 `outputs/2026-10-01-16/stage1_cached_dual_boundaryblend035_overlap2_latest_124/`：
8 steps/chunk、双 anchor、blend `0.35`、前两帧衰减 soft overlap。统计为 MAD `3.665/13`、
边缘差 `2.274`，teacher 为 `4.295/15`、`2.226`。该 overlap 使接触帧不再出现整段场景
的硬瞬移，但 frame 85、102 一带仍有 temporal VAE ghosting；更低 MAD 不能单独证明质量
更好。因此它是当前 Stage1-style 可行性主候选，不是质量等价或 Stage2 Demo。

本时段还完成：

- `outputs/README.md` 增加 2026-10-01-15/16/17 的索引和负结果说明；
- `README.md`、`docs/causal_prototype_report.md` 同步记录 overlap=2 主候选和 mixed-history
  负结果；
- 原始日志已放入 `outputs/2026-10-01-17/`，没有继续堆在 outputs 根目录；
- 新增 `history-mix` 后，`.venvs/h3world/bin/pytest -q H3-World/tests` 为 **16 passed**，
  `py_compile H3-World/code/causal/*.py` 通过。

下一步主线仍然是：如果要继续提高 Stage1，必须使用多 clip 的 generated-history unroll、
更接近 SolarWM AnyFlow 的 flow-map 训练和明确的 temporal continuity objective；不能把
soft overlap、causal mask、KV cache 或普通 8-step solver 称为 Stage2。当前用户提出的
“上一 chunk 最后一帧 + 当前 action”已经有 latent dual 和 RGB-prefix dual 两条可运行路径，
但单场景原型还没有证明它能稳定改善长片质量。

## 2026-10-01 09–10 时段：扩大 causal 适配容量，生成无硬瞬移主 Demo

mixed-history 结果表明，直接把一条旧 generated rollout 混入单场景训练不能解决漂移。本时段
改做架构容量对照：保持同一 seed、prompt、首帧、124 帧 teacher、双 anchor、persistent
cached-history、H3 scheduler `shift=2.22` 和 8 steps/chunk，只把尾部 QKV LoRA 从 4 个
DiT block 扩展到 8/16 个 block。没有修改随机种子，没有读取未来 chunk，也没有用输出后处理
作为唯一修复。

| 配置 | 可训练参数 | train loss | validation loss | replay | 124 帧 rollout MAD/p95 | edge diff |
|---|---:|---:|---:|---:|---:|---:|
| tail4（此前主候选） | `860,160` | `0.10318→0.08088` | `0.10897→0.09149` | `0` | `4.277/16` | `2.275` |
| tail8 | `1,720,320` | `0.10087→0.06886` | `0.10414→0.08553` | `0` | `4.508/16` | `3.157` |
| tail16 | `3,440,640` | `0.10087→0.06384` | `0.10414→0.08219` | `0` | `4.786/18` | `2.563` |

原始产物：

```text
outputs/2026-10-01-18/stage1_clean_dual_tail8/
outputs/2026-10-01-18/stage1_clean_dual_tail16/
outputs/2026-10-01-18/rollout_tail8_124/
outputs/2026-10-01-18/rollout_tail16_124/
```

tail16 的并排主 Demo 已放到 outputs 根目录：

```text
H3-World/outputs/h3world_30step_vs_stage1_tail16_dual_anchor_8step_124.mp4
```

抽帧检查 frame 17、34、51、68、85、102、119：tail16 右侧的停车场和人物轨迹在 chunk
边界处保持连续，没有 tail4 + soft overlap 版本那种大范围 temporal-VAE ghosting 或整段
场景瞬时切换。tail16 的长期路径仍会逐渐偏离 30-step teacher（例如后段进入不同的坡道
方向），所以当前结论是“causal chunk 的硬瞬移问题已通过扩大适配容量显著缓解”，不是
“teacher 质量已经复现”。

对同一 tail16 adapter 另跑了 `blend=0.2`、overlap=1 的后处理对照：MAD `5.206/20`、edge
`2.923`，比无后处理 tail16 的 `4.786/18`、`2.563` 更模糊，因此没有把它设为主 Demo。
2026-10-01-16 的 blend=0.35、overlap=2 仍保留用于边界约束说明，不取代 tail16 架构版本。

文档已同步更新：根目录 README 现在把 tail16 并排视频列为当前主 Demo，并明确说明它解决的是
chunk 边界硬切换而非长期语义 drift；`docs/causal_prototype_report.md` 和
`H3-World/outputs/README.md` 均加入 tail8/tail16 对照、指标和负结果。

## 2026-10-01 09–10 时段：tail16 的 RGB-prefix image anchor 复测

为直接验证用户提出的“上一 chunk 最后一帧作为图片条件”，在 tail16 adapter 上使用
`dynamic_last_frame_rgb_dual` 做了完整 124 帧训练和 rollout。slot 1 的条件严格走：当前
prefix temporal VAE decode → 取最后可见 RGB frame → `encode_video(process_image=True)` →
patchify；slot 0 仍为原始首帧，当前 action 绑定不变。

训练目录：

```text
outputs/2026-10-01-19/stage1_rgb_dual_tail16/
```

配置为 124 帧、8 chunks、末 16 blocks、rank-8 QKV LoRA、160 步、shift=2.22。训练结果：

```text
train loss:       0.09810 -> 0.06211
validation loss:  0.10255 -> 0.08152
replay max error:  0
trainable params: 3,440,640
```

rollout：

```text
outputs/2026-10-01-19/rollout_rgb_tail16_124/cached.mp4
```

统计为 MAD `4.585/17`、空间边缘差 `2.368`，比 latent tail16 的 `4.786/18`、`2.563` 更接近
teacher `4.295/15`、`2.226`。采样耗时 `567.41 s`，latent tail16 为 `304.52 s`；7 次
RGB-prefix anchor 处理约 `202.8 s`，CPU KV 峰值约 `13.2 GiB`。根目录新增并排视频：

```text
H3-World/outputs/h3world_30step_vs_stage1_rgb_tail16_dual_anchor_8step_124.mp4
```

该结果说明“上一末帧 + action”在 H3 原生图片分支中可训练、可自由 rollout，且连续性指标有
轻微改善；但代价是几乎翻倍的采样时间，因此效率主 Demo 仍采用 latent tail16，RGB tail16
作为用户方案的直接对照。

## 2026-10-01 10 时段：243 帧（10.125 秒）长时段验证

为检查 5 秒之后的行为，使用相同 seed 13、双 latent anchor、tail16 adapter、H3 shift=2.22
和 8 steps/chunk 生成了 243 帧 causal 视频：

```text
outputs/2026-10-01-10/tail16_dual_243_verified/cached.mp4
```

它包含 15 个 chunk、120 次 noisy denoiser forward、15 次 clean KV commit，采样
`633.52 s`，GPU allocated 峰值约 `39.0 GiB`，CPU raw-KV 峰值约 `13.2 GiB`。视频长
`10.125 s`，灰度 MAD 为 `5.044/19`，空间边缘差 `3.400`，最大相邻帧 MAD 约 `9.73`；
抽帧和周期性边界检查显示没有整段场景瞬时切换，但最后几秒出现明显的停车场几何 drift。
这证明 causal 分块和 CPU KV cache 可以把序列延长到 10 秒以上，不代表长时段质量已经通过。

同一 seed、相同 prompt、首帧和 resolution 的 H3 30-step full-sequence baseline 也尝试生成
243 帧，但在第一次 denoiser 调用申请额外 874 MiB 时 OOM（44.4-GiB L40 只剩约 579 MiB）。
失败记录在：

```text
outputs/2026-10-01-10/baseline30_243_verified/
```

该 OOM 是显存边界证据，不能作为画质对照；它同时说明当前分块 + CPU raw-KV 路径对长视频
有实际工程价值。124 帧 tail16 仍然是面试主 Demo，243 帧作为长时段和显存报告附件。

官方 SolarWM 对照路径也已核对并写入根 README：MiniMax-H3 的
`stage0p5.py`、`stage1.py`、`stage2.py` 以及对应 YAML 配置分别明确了 bidirectional FM、
`causal_mode: teacher_forcing` + `objective: anyflow_forward_map`、以及
`causal_mode: self_gradient_forcing` + frozen teacher/trainable critic。当前 H3 原型只声称
Stage1-style causal teacher-forcing，不把普通 flow loss、KV cache 或 8-step solver 称作
AnyFlow/Stage2。

## 2026-10-01 20 时段：固定 action intervention 已完成

本时段完成面试题最后一个关键验收：验证 SolarWM-style causalization 是否保留 H3-World
原有的 action interface。benchmark 新增 `--action-preset`，支持 W/S/A/D 别名，分别解析为
H3 的 `forward`、`back`、`strafe-left`、`strafe-right`，并把 action preset 写入每个
`setup.json` 和运行 JSON。没有修改随机种子。

固定条件：同一首帧、同一 prompt、seed `13`、832×480、124 RGB frames（5.17 秒）、同一
H3 action LoRA、同一初始 video/audio noise。对 W/S/A/D 各跑了两条完整视频：

1. 原始 H3-World：30 full-sequence denoiser steps，fixed first-frame image condition；
2. causal Stage1-style：tail16 rank-8 QKV LoRA、5 latent-frame chunk、5-chunk sliding
   history、persistent per-layer raw KV、CPU offload、latent dual anchor、8 steps/chunk。

四个 causal case 均完成 `64` 次 noisy denoiser forward、`8` 次 clean commit，adapter replay
 误差仍为 `0`；采样时间为 W/S/A/D=`342.7/347.5/348.5/344.5 s`，GPU allocated peak
 `38.6–39.0 GiB`，CPU raw-KV peak `13.2 GiB`。原始 30-step case 的采样时间为
 `359.2/362.3/373.3/371.2 s`，没有 KV cache。因而这组实验不是把 64 次 forward 声称成
 8 次 denoiser；它仍然是 8 steps/chunk × 8 chunks，并通过短 active sequence 与历史 KV
 复用获得因果分块效率。

新增产物：

```text
H3-World/outputs/2026-10-01-20/action_W_baseline30_124/baseline.mp4
H3-World/outputs/2026-10-01-20/action_S_baseline30_124/baseline.mp4
H3-World/outputs/2026-10-01-20/action_A_baseline30_124/baseline.mp4
H3-World/outputs/2026-10-01-20/action_D_baseline30_124/baseline.mp4
H3-World/outputs/2026-10-01-20/action_W_tail16_124/cached.mp4
H3-World/outputs/2026-10-01-20/action_S_tail16_124/cached.mp4
H3-World/outputs/2026-10-01-20/action_A_tail16_124/cached.mp4
H3-World/outputs/2026-10-01-20/action_D_tail16_124/cached.mp4
H3-World/outputs/h3world_stage1_tail16_action_intervention_grid_124.mp4
```

四行 4×2 review grid 的左列是 Original H3-World、右列是 causal Stage1-style，行顺序为
W/S/A/D。`action_summary.json` 统一记录 inference time、GPU peak、CPU KV、denoiser
forwards、clean commits、frame MAD、boundary MAD 和 action response。causal 124 帧 MAD
均值/p95 分别为：W=`4.785/18`、S=`4.914/19`、A=`4.925/19`、D=`4.803/18`；RGB
boundary score（帧 17、34、51、68、85、102、119 的相邻帧 MAD 均值）分别为
`5.69/6.14/6.00/5.72`。这些是运动/连续性描述，不是视频质量分数。

action intervention 结论：W/S causal 视频均值 RGB 差异 `7.50`，A/D 差异 `6.97`，接触图中
人物和停车场轨迹确实随 action 改变，说明 action 文本行和 packed action rows 没有因 causal
mask/KV cache 而被抹掉。原始 H3 的全画面 Farneback horizontal flow 对 A/D 给出
A=`+1.08`、D=`−1.60` 的明显方向差；causal 全画面 flow 受背景和长期 drift 影响，方向
proxy 不够稳定，所以报告为 action-response 辅助证据，不能夸大为严格 action accuracy。

需要明确的限制：tail16 adapter 的训练 teacher 是单场景 W action clip，这次 S/A/D 是同一
checkpoint 的跨 action robustness/intervention，不是多 action causal fine-tuning。当前结果
已经足够支持面试题的最小结论：H3-World 可以做 SolarWM-style causal chunk rollout，且 action
interface 仍然有效；剩余主要问题是 generated-history distribution shift 和长期 drift，
不是 action binding 消失。后续不再扩展 SGF/DMD、更多 anchor 或新的 fancy metric；最终 demo
固定为主 tail16 causal 视频、RGB-anchor ablation、generated-history mix 负对照，action grid
作为 action-control 证据。

本时段新增代码：

- `H3-World/code/causal/evaluate_action_control.py`：中心区域 Farneback optical-flow
  action-response proxy，明确输出方向约定和局限；
- `H3-World/code/causal/make_action_grid.py`：生成 W/S/A/D 四行 Original-vs-Causal 网格；
- `H3-World/code/causal/benchmark.py`：`--action-preset W/S/A/D` 和 action metadata。

验证：四 action benchmark 均 `status=complete`、每个 MP4 124 帧；
`.venvs/h3world/bin/python -m py_compile H3-World/code/causal/*.py` 通过。已有测试仍为
`16 passed`，没有残留 benchmark 进程。

## 2026-10-02 00 时段：多 action teacher-forcing 与 action-control 复核

上一节的单 W teacher adapter 只能说明 action 行没有被 causal mask 完全抹掉，不能说明
W/S/A/D 的方向控制仍然准确。本时段补齐了四个固定条件的原始 H3 teacher clip，并训练
了一个共享的 multi-action causal adapter。四个 teacher 都使用 seed `13`、同一首帧、同一
prompt、同一初始 video/audio noise 和 124 帧；只改变 H3 action：W=`forward`、S=`back`、
A=`strafe-left`、D=`strafe-right`。

### shared adapter 训练

四个 teacher latent 位于：

```text
outputs/2026-10-01-21/action_W_teacher_latents/
outputs/2026-10-01-21/action_S_teacher_latents/
outputs/2026-10-01-21/action_A_teacher_latents/
outputs/2026-10-01-21/action_D_teacher_latents/
```

tail8 训练成功，目录为 `outputs/2026-10-01-21/stage1_multiaction_tail8/`；它有
1,720,320 个可训练参数，train `0.1588316→0.1419430`，validation
`0.1908010→0.1823134`，replay error `0`。它的四个固定 action 和 W:3,A:2,D:3、
D:2,S:3,A:3 schedule rollout 已经完整结束。

为检验容量限制，使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在 GPU1 重跑
末 16 个 DiT block。目录为：

```text
outputs/2026-10-02-00/stage1_multiaction_tail16_retry/adapter.pt
```

这次不是 OOM：adapter 已保存且格式为 `h3_causal_tail_qkv_v2`，16 个 block、
3,440,640 个可训练参数，replay error `0`，train `0.1588316→0.1385778`，validation
`0.1908010→0.1801894`，训练 allocated peak 约 `40.86 GiB`。之前
`outputs/2026-10-01-21/stage1_multiaction_tail16/training.json` 的 OOM 记录仍保留，说明
该配置需要 allocator 处理和空闲显存；不能把那次失败解释为算法失败。

### tail8 固定 action 结果

tail8 rollout 目录为 `outputs/2026-10-01-21/action_*_multiaction_tail8/`。每条视频均为
124 帧、5.17 秒、64 次 noisy denoiser forward、8 次 clean KV commit，GPU allocated
峰值约 `39.2 GiB`，CPU raw-KV 峰值 `13.19 GiB`。中心区域 Farneback 水平 flow proxy 为：

| Action | Original H3 flow | Causal tail8 flow | Causal gray MAD | Causal boundary RGB MAD |
|---|---:|---:|---:|---:|
| W | `-1.111` | `-0.751` | `3.904` | `6.010` |
| S | `-1.084` | `-0.671` | `3.859` | `6.142` |
| A | `+1.077` | `-0.575` | `3.838` | `5.891` |
| D | `-1.602` | `-0.676` | `3.839` | `6.029` |

tail8 causal 的 W/S 和 A/D mean RGB 差异分别为 `3.80` 和 `7.05`。因此 action 条件会
改变输出，但水平 flow 仍然同号；不能写成 W/S、A/D 方向控制已经保真。

### tail16 shared adapter 结果

tail16 四条固定 action 和两个 schedule 已经在 GPU0/1/2/3/4/6 完整完成，目录为：

```text
outputs/2026-10-02-00/action_W_multiaction_tail16_retry/
outputs/2026-10-02-00/action_S_multiaction_tail16_retry/
outputs/2026-10-02-00/action_A_multiaction_tail16_retry/
outputs/2026-10-02-00/action_D_multiaction_tail16_retry/
outputs/2026-10-02-00/action_schedule_WAD_multiaction_tail16_retry/
outputs/2026-10-02-00/action_schedule_DSA_multiaction_tail16_retry/
```

每条仍是 `124` 帧、`64 + 8` 次 noisy/commit、CPU KV `13.19 GiB`，GPU allocated 峰值
约 `38.6–39.0 GiB`。新的 4×2 对比视频为：

```text
outputs/h3world_stage1_multiaction_tail16_action_grid_124.mp4
```

统一指标表和 JSON 为：

```text
outputs/2026-10-02-00/action_multiaction_tail16_summary.md
outputs/2026-10-02-00/action_multiaction_tail16_summary.json
```

tail16 的全画面 flow proxy 为 W=`-0.540`、S=`-0.623`、A=`-0.594`、D=`-0.579`；W/S
和 A/D 的 causal mean RGB 差异分别只有 `4.10` 和 `3.56`。相比原始 H3 的 A=`+1.077`
和 D=`-1.602`，方向分离没有恢复，扩大 QKV LoRA 容量也不能解决这个问题。causal MAD
约为 `3.18–3.38`，低于 teacher 不能当作质量更好；它可能表示运动减弱或生成漂移。

### chunk-level schedule 结果

W:3,A:2,D:3 的 tail16 逐段 flow proxy（W=`0:51`、A=`51:85`、D=`85:124`）为：

```text
W -0.746, A -0.603, D -0.214
```

原始 H3 对应为 `-0.033, +1.564, -1.278`。D:2,S:3,A:3 的 tail16 逐段 flow（D=`0:34`、
S=`34:85`、A=`85:124`）为：

```text
D -0.823, S -0.648, A -0.302
```

原始 H3 对应为 `-1.212, -1.635, -1.210`。新增带输入动作标注的并排 schedule 视频：

```text
outputs/h3world_schedule_WAD_multiaction_tail16_124.mp4
outputs/h3world_schedule_DSA_multiaction_tail16_124.mp4
```

视频中的 action label 是输入时间表，不是模型预测结果；这些 schedule 证明 action rows
确实可以按 chunk 进入 causal pipeline，但当前 Stage1 clean-history 训练没有维持原始 H3
的方向响应。

### 当前结论边界

多 action teacher-forcing 比单 W robustness intervention 更接近题目要求，但 tail8 和
tail16 都得到同一个结论：

1. causal mask、persistent raw KV、clean commit、124 帧 rollout 和 action schedule 机械链路可运行；
2. 变更 action 会改变 causal 输出，说明 action conditioning 没有完全消失；
3. W/S/A/D 的方向性以及 chunk-level 方向切换没有保真，不能宣称“action control 完全保留”；
4. 主要剩余问题是 generated-history distribution shift、单场景训练容量和 Stage1 目标本身，
   而不是把 KV cache 再扩展一倍；
5. 这组负结果反而说明为什么 SolarWM 需要 Stage2 self-gradient forcing / distillation。

因此面试报告的最终表述应改为：

> We successfully causalized H3-World with chunk-wise attention and persistent KV caching.
> Multi-action clean-history teacher forcing keeps the action rows active and produces
> action-dependent videos, but the current Stage1 prototype does not yet preserve the
> original H3 directional action response under generated-history rollout. Stable action
> grounding and few-step long-horizon quality require generated-history distribution matching,
> motivating SolarWM Stage2-style training.

本时段新增/修改：`train_pretrained_multiaction.py` 已用于四 action shared adapter，新增
`summarize_action_experiment.py` 统一 timing/memory/MAD/boundary/flow 表，
`make_side_by_side.py` 支持在 schedule 视频中标出每帧输入 action。README、
`docs/causal_prototype_report.md`、`H3-World/outputs/README.md` 和本文件均已同步这个
结论。最后验证已完成：所有 6 条 tail16 rollout 为 `status=complete`、124 帧、64 noisy
forwards、8 clean commits；所有 causal Python 文件通过 `py_compile`，测试为 `16 passed`。
不再增加 anchor trick 或 SGF/DMD；Stage1 prototype 可以收工，后续若继续应进入
generated-history distribution matching / SolarWM Stage2，而不是继续扩展当前诊断模块。

## 2026-10-02 04 时段：action fidelity decomposition 与保守 adapter 修复

根据建议文件 `pasted-text-1.txt`，本时段不再把“动作失真”归因于单一模块，补做了
causal function shift 与 generated-history shift 的分解。

### 39 帧、30 steps/chunk、无 adapter 的 generated-history 对照

新增目录：

```text
outputs/2026-10-02-04/action_A_base_generated30_39/
outputs/2026-10-02-04/action_D_base_generated30_39/
```

两条运行使用同一首帧、prompt、seed=13、同一初始噪声、dynamic dual latent anchor，
只把历史来源设为模型生成结果；每条都是 39 RGB frames、12 latent frames、3 chunks、
90 次 noisy denoiser forwards 和 3 次 clean commit。结果为：

```text
A horizontal flow: -0.136920
D horizontal flow: -0.152110
A-D difference:     +0.015190
```

此前 clean teacher history、无 adapter、30 steps/chunk 的 A/D 结果为
`+0.390690/-0.564994`，差值 `+0.955683`。因此动作方向在短片段的 causal mask 阶段
仍保留一部分，但换成 generated history 后已经明显塌缩；这比单纯增加 solver steps
更直接地支持 generated-history distribution shift 是主要问题之一。

合并诊断证据如下：

| 条件 | A flow | D flow | A-D |
|---|---:|---:|---:|
| 原始 H3 teacher，39f，30 steps | +1.181253 | -0.842124 | +2.023377 |
| causal，无 adapter，clean history，39f | +0.390690 | -0.564994 | +0.955683 |
| causal，普通 MSE tail16，clean history，39f | -0.572779 | -0.877738 | +0.304958 |
| causal，pair-loss tail8，clean history，39f | -0.346916 | -0.614203 | +0.267288 |
| causal，无 adapter，generated history，39f，30 steps | -0.136920 | -0.152110 | +0.015190 |
| causal，无 adapter，generated history，124f，8 steps | -0.017959 | -0.025754 | +0.007795 |

水平光流仍只是动作响应 proxy；A/D 比较最适合该指标，W/S 需要结合人物位移、尺度或
深度方向 proxy，不能要求四个动作在水平光流上完全对称。

### 新增 base-output regularization

修改：`code/causal/train_pretrained_multiaction.py`。

新增参数：

```text
--base-output-reg-weight FLOAT
```

该项约束训练后的 causal adapter 输出不要偏离零 adapter causal 基线：

```python
L = L_flow + lambda_base * MSE(adapter_output, zero_adapter_causal_output)
```

它在输出空间约束动作几何，比单纯惩罚 LoRA 参数更直接；同时保留 action-pair loss
来约束 W/S、A/D 的差分。新增训练任务：

```text
outputs/2026-10-02-04/stage1_multiaction_generated_reg_tail8_lr1e4/
```

配置为 generated-history、tail8、rank8、lr=1e-4、base-output-reg=0.5、
action-pair=0.5，运行在 GPU2。它与 GPU0 的 generated-history pair-loss 训练、GPU1 的
schedule-aware clean-history 训练并行，不打断已有任务。

### 历史状态（已被后续 09 时实验覆盖）

当时记录的训练任务：

```text
GPU0 stage1_multiaction_generated_pair_tail8
GPU1 stage1_multiaction_schedule_pair_tail8
GPU2 stage1_multiaction_generated_reg_tail8_lr1e4
```

GPU3/GPU4/GPU6 已用于补齐 39 帧 generated-history 30-step action decomposition；A/D 已
完成，W/S 正在完成。训练完成后必须分别复测 clean/generated 39f 和 generated 124f，
再决定哪个 checkpoint 用于最终 W/S/A/D action grid。不能仅凭 train/validation flow
loss 低就宣称 action fidelity 恢复。

本时段验证：

```text
.venvs/h3world/bin/python -m py_compile \
  H3-World/code/causal/train_pretrained_multiaction.py
```

通过；这些任务状态属于当时记录，后续 09 时实验已经覆盖本轮动作残差诊断。

### 2026-10-02 09 时：动作残差隔离实验

上一节中的 GPU0/GPU1/GPU2 训练状态是旧记录，不能作为当前状态；本节更新到本次
动作保真拆解后的实际结果。

`stage1_action_residual_only39` 已完成：只训练 action-dependent Q/K/V residual，冻结
通用 QKV adapter，8 个尾部 DiT block，39 帧、3 chunks、clean-history teacher forcing。
训练 metadata 为 train `0.2411168 -> 0.2383919`、validation `0.2668816 -> 0.2653980`、
`replay_max_error=0`。在相同 seed、首帧和 teacher latent、30 steps/chunk 下：

| 条件 | A flow | D flow | A-D |
|---|---:|---:|---:|
| causal 无 adapter | `+0.390690` | `-0.564994` | `+0.955683` |
| causal action QKV residual（冻结通用 QKV） | `+0.195898` | `-0.619446` | `+0.815343` |

该 residual 保留了 A/D 的相反符号，但没有恢复原始 H3 teacher 的 `+2.023377` 差异，且
A 响应反而减弱。因此“在当前 chunk 加一个动作 residual”不能等价于恢复原始全序列
attention 的 action geometry。

`stage1_action_hidden_only39` 也已完成训练：只训练 hidden FiLM residual、冻结通用
QKV，8 个尾部 block，`replay_max_error=0`；训练 train `0.2411168 -> 0.2411083`、
validation `0.2668816 -> 0.2647181`，目前正在等待其 A/D benchmark 完成，不能提前把它
当作修复结果。

该 benchmark 现已完成，输出为：

```text
outputs/2026-10-02-08/actionhidden_only_A_clean30_39/
outputs/2026-10-02-08/actionhidden_only_D_clean30_39/
outputs/2026-10-02-08/actionhidden_only_clean30_39_flow.json
```

hidden-only 的 A=`+0.313006`、D=`-0.658585`、A-D=`+0.971591`。它比 qkv-only 的
`+0.815343` 稍稳定，也保留了 A/D 相反符号，但仍明显低于原始 H3 teacher 的
`+2.023377`，所以不能作为“四方向完全保真”的 checkpoint。

### 当前诊断结论

动作保真损失不是单个开关造成的：

1. causal mask 改变了 H3 原 action LoRA 所适应的全序列交互图；缓存路径中 action prefix
   不再和当前 video hidden 做同样的双向层间反馈。
2. raw KV 在每个 chunk 的 clean sigma=0 状态提交，后面的 noisy solver 读取固定历史；
   原始 H3 则在每个 denoising 时刻重算整段历史，两个 score field 不同。
3. adapter 的主要训练协议是 clean teacher forcing，而自由 rollout 将生成误差写入下一
   chunk 的 KV 和 latent anchor；39 帧 generated-history 已把 A-D 从 `0.956` 降到
   `0.015`，所以仅增加 solver steps 或 action residual 不能解决长期动作塌缩。
4. 共享 MSE/QKV 目标更容易学习四个动作共同的场景和外观，action residual 的 pair loss
   只能部分拉开 A/D，不能自动学习 schedule transition 或 generated-history 分布。
5. latent dual anchor 是 temporal latent 的 image-like 近似，不等价于 H3 原生 RGB image
   condition；它改善 handoff 时也可能把场景先验压过动作残差。

另外，`action_prefix_mode=all` 的上界诊断并没有恢复动作：39 帧 generated-history 的
A-D=`-0.010542`。因此问题不能简化为“当前 chunk 没看到 action token”。W/S 也不能用
水平 flow 符号验收，原始 H3 的前进/后退本身可能产生同号的相机/深度运动；A/D 才适合
用水平 flow 直接判断左右。

### 2026-10-02 10 时：generated-history 与 schedule-aware 修复实验

根据建议文件的优先级，开始训练真正使用 causal generated history 的 action adapter，
而不是继续只用 clean teacher history。已有的四个 124 帧固定动作 teacher latent 位于：

```text
outputs/2026-10-01-21/action_{W,S,A,D}_teacher_latents/baseline_latents.pt
```

对应的 124 帧、8-step/chunk、无 adapter causal generated latent 位于：

```text
outputs/2026-10-02-04/action_{W,S,A,D}_base_causal124_latents/cached_latents.pt
```

新增训练任务：

```text
outputs/2026-10-02-10/stage1_action_generated_hidden124/
```

它只训练 8 个尾部 block 的 hidden action residual，冻结通用 QKV，使用 generated-history
target、A/D 与 W/S pair loss 和弱 base-output regularization，目标 chunk 覆盖完整 37 个
latent frames（124 RGB frames）。

同时补充了 `train_pretrained_multiaction.py --action-specs`。它现在能接受：

```text
W:3,A:2,D:3
D:2,S:3,A:3
```

并把 schedule 展开为每个 latent interval 的 action one-hot，而不是把一个 action 粗略地
广播给整段视频。该解析已经用 37 latent frames 的 one-hot 检查验证。

为训练 schedule-aware residual，先生成了真实的 124 帧 generated-history：

```text
outputs/2026-10-02-10/action_schedule_WAD_generated124/
outputs/2026-10-02-10/action_schedule_DSA_generated124/
```

两条 rollout 都是 124 frames、8 chunks、64 noisy forwards、8 clean commits，CPU KV 峰值
约 13.19 GiB。对应的 schedule hidden residual 训练为：

```text
outputs/2026-10-02-10/stage1_schedule_generated_hidden124/
```

此外启动了 39 帧快速诊断和一个强 action-pair loss 变体：

```text
outputs/2026-10-02-10/stage1_action_generated_hidden39/
outputs/2026-10-02-10/stage1_action_generated_hidden39_pairstrong/
```

### 2026-10-02 10 时训练结果（均已完成）

五项训练的最终状态如下。所有 adapter 均采用 8 个尾部 DiT block、rank 8、`action_residual_mode=hidden`、`dual_anchor_protocol`，可训练参数 774,144。

| 实验 | 帧数 | 步数 | history 类型 | A/D pair loss | 训练完成 | wall 时间 | 峰值显存 |
|---|---:|---:|---|---:|---|---:|---:|
| `stage1_action_generated_hidden39` | 39 | 160 | generated | 4.0 / 8.0 | ✅ | 823 s | 25.4 GiB |
| `stage1_action_generated_hidden39_pairstrong` | 39 | 160 | generated | 强化版 | ✅ | 823 s | — |
| `stage1_action_generated_feedback_hidden39` | 39 | — | generated | — | ✅ | 926 s | — |
| `stage1_schedule_generated_hidden124` | 124 | 160 | generated | 无 pair | ✅ | 1165 s | 20.8 GiB |
| `stage1_action_generated_hidden124` | 124 | 160 | generated | 4.0 / 8.0 | 🔄 运行中 | — | — |

`stage1_action_generated_hidden124` 截至本记录时仍在运行（`status: running`），特征提取已完成（768 s），进入训练阶段，尚无最终 checkpoint。

### 2026-10-02 10 时 cached inference 评估

对上述已完成的 adapter 进行了 7 次 cached inference 评估，均 `status: complete`，使用 `attention_backend: PyTorch SDPA`，`trained_for_causal: false`：

| 实验 | 帧数 | 步数 | history | action_feedback | adapter | 采样时间 | VRAM 峰值 | KV cache 峰值 |
|---|---:|---:|---|---|---|---:|---:|---:|
| `action_feedback_A_clean30_39` | 39 | 30 | clean | True | 无 | 322 s | 33.5 GiB | 6.5 GiB |
| `action_feedback_D_clean30_39` | 39 | 30 | clean | True | 无 | — | — | — |
| `action_generated_hidden39_eval/A` | 39 | 30 | generated | False | `hidden39` | 357 s | 39.6 GiB | 6.5 GiB |
| `action_generated_hidden39_eval/D` | 39 | 30 | generated | False | `hidden39` | — | — | — |
| `pairstrong_hidden39_eval/A` | 39 | 30 | generated | False | `pairstrong` | 355 s | 39.6 GiB | 6.5 GiB |
| `pairstrong_hidden39_eval/D` | 39 | 30 | generated | False | `pairstrong` | — | — | — |
| `schedule_generated_hidden124_eval_DSA` | 124 | 8 | generated | False | `schedule_hidden124` | 336 s | 39.9 GiB | 13.5 GiB |
| `schedule_generated_hidden124_eval_WAD` | 124 | 8 | generated | False | `schedule_hidden124` | — | — | — |

**A-D horizontal flow 对比**（`evaluate_action_control.py` / `evaluate_action_schedule.py` 输出）：

| 条件 | A flow x | D flow x | A-D |
|---|---:|---:|---:|
| clean-history，无 adapter，clean30，39 帧 | +0.392 | −0.603 | **+0.994** |
| generated-history，`hidden39` adapter，30 steps | −0.221 | − | — |

`action_feedback_clean30_39` 的 clean-history A-D 差值 +0.994 是本轮最好的短时方向分离结果，优于上一轮的 +0.972（2026-10-02-08）。但 `action_generated_hidden39_eval` 的 generated-history A 水平流为 −0.221，接近零，继续确认 generated-history 分布偏移导致动作塌缩的结论。

这进一步验证：clean-history 短时 rollout 能保留方向响应，free-running generated history 无论加什么 generated-history adapter，动作方向仍会在 39 帧内接近消失。修复路径仍指向 generated-history self-rollout 训练分布匹配（Stage2-style），而不是继续堆叠 adapter 变体。

---

## 2026-10-02 23:44 - Task 2: Scheduled-Sampling 实验启动

### 实验设计

为验证 `--history-mix-schedule` 修复 generated-history 动作塌缩的效果，启动了 3 组并行对比实验（39 帧，W/A/D 三动作）：

| 实验 | GPU | 配置 | 目的 |
|---|---|---|---|
| baseline_clean_only | 2 | 无 history-latent-dirs，纯 clean history | 对照组，验证 clean-history 性能上界 |
| fixed_mix_0.5 | 3 | `--history-mix 0.5`，固定 50% generated history | 验证固定混合比例效果 |
| curriculum_0.0_to_0.5 | 4 | `--history-mix-schedule 0.0:0.5` | 验证课程学习效果（核心方案） |

### 实验参数

```bash
--steps 160
--tail-blocks 8
--rank 8
--lr 1e-3
--anchor-mode dynamic_last_frame_dual
--history-protocol cached
--scheduler-steps 8
--scheduler-shift 2.22
--action-residual
--action-residual-mode hidden
--no-qkv-adapter
```

### 数据路径

- **Teacher latents**: `outputs/2026-10-02-03/action_{W,A,D}_teacher_39/`
- **Generated-history latents**: `outputs/2026-10-02-10/generated_history39/action_{W,A,D}_generated39/`
- **输出目录**: `outputs/2026-10-02-scheduled-sampling/`

### 进程状态

- baseline PID 1369208 (GPU 2)
- fixed_mix PID 1371090 (GPU 3)  
- curriculum PID 1372925 (GPU 4)

所有进程已在 23:44 启动，当前处于特征提取阶段。

### 监控工具

创建了两个脚本：
1. `monitor_scheduled_sampling.sh` - 手动检查实验进度
2. `wait_and_benchmark.sh` - 自动等待完成并运行 benchmark

### 验收标准

预期：
- **baseline (clean only)**: A-D flow ≈ +0.956（已知上界）
- **fixed_mix 0.5**: A-D flow > 0.015（如果有效）
- **curriculum 0.0→0.5**: A-D flow ≥ fixed_mix（课程学习应该更好或至少持平）

如果 curriculum 在 generated-history rollout 下能将 A-D 从 +0.015 提升到 >0.1，则确认 scheduled-sampling 有效，可以扩展到 124 帧。


## 2026-10-03 02 时：Scheduled-Sampling 公平 A/D 评估完成

上一轮的三个 scheduled-sampling checkpoint 已经用同一套推理条件重新评估，避免把不同 action、不同 chunk 或不同 step 数的单条 benchmark 误当成动作对照。评估配置为：

```text
seed=13；39 RGB frames（13 latent frames）
causal chunk=5 latent frames；3 chunks；history_chunks=5
30 denoiser steps/chunk；flow_shift=2.22
history_source=generated；anchor_mode=dynamic_last_frame_dual
action_prefix_mode=own；action_feedback=false；cache_device=cpu
每条视频 denoiser forwards=90，clean commits=3
```

所有 A/D 使用相同首帧、prompt、seed 和初始 noise，只替换 action；输出集中在：

```text
outputs/2026-10-03-02/scheduled_sampling_fair39_30steps/
```

### A/D horizontal flow 结果

| checkpoint | A mean flow | D mean flow | A-D | sampling time（A/D） | peak GPU | CPU KV peak |
|---|---:|---:|---:|---:|---:|---:|
| baseline_clean_only | −0.184 | −1.029 | **0.845** | 310/358 s | 39,924 MiB | 6,484 MiB |
| fixed_mix_0.5 | −0.412 | −1.415 | **1.003** | 350/349 s | 39,924 MiB | 6,484 MiB |
| curriculum_0.0_to_0.5 | −0.379 | −1.265 | **0.886** | 389/359 s | 39,924 MiB | 6,484 MiB |
| 原始 H3 teacher（39 帧） | +1.181 | −0.842 | **2.023** | — | — | — |
| causal 无 adapter（39 帧 generated） | −0.137 | −0.152 | **0.015** | — | — | — |

三组 adapter 都明显超过无 adapter generated-history 的动作塌缩；fixed 50% mix 当前短时 A/D 分离最好，但仍只有原始 H3 teacher 的约一半，且 A 的绝对方向仍受 generated-history 漂移影响。三个模型不是用不同 action 比较，而是在同一个 checkpoint 内分别生成 A 和 D；原始完整 JSON 分别位于各 checkpoint 目录的 `AD_flow.json`，每条视频的 `cached.json` 记录完整耗时、峰值显存、KV 峰值和 90 次 denoiser forwards。

这一步支持的结论是：scheduled-history 训练能恢复一部分 action sensitivity，但还不能宣称四方向保真或已经完成 SolarWM Stage2。下一步将选择 `fixed_mix_0.5/action_adapter.pt` 作为当前最佳诊断 checkpoint，执行统一的 W/S/A/D、124 frames、8 steps/chunk rollout；若长时动作仍衰减，将把结果报告为 generated-history distribution shift，而不是继续堆叠普通 adapter。

## 2026-10-03 03 时：固定 scheduled-sampling checkpoint 的 124 帧 W/S/A/D 验收

根据 39 帧公平 A/D 诊断，选择当前短时方向分离最好的 `fixed_mix_0.5` action residual checkpoint 做最终长视频验收：

```text
outputs/2026-10-02-scheduled-sampling/fixed_mix_0.5/action_adapter.pt
```

四个 action 使用同一张首帧、prompt、seed=13、初始 video/audio noise 和同一 checkpoint，只改变 W/S/A/D。统一配置：

```text
124 RGB frames = 5.17 s；37 latent frames；8 causal chunks
5 latent-frame chunk；history_chunks=5；generated history
8 denoiser steps/chunk；64 noisy forwards + 8 clean KV commits
flow_shift=2.22；dynamic_last_frame_dual；action_prefix_mode=own
action_feedback=false；persistent raw KV on CPU
```

原始 H3 对照是已有的同 seed、同首帧、同 prompt 的 30-step full-sequence 视频：

```text
outputs/2026-10-01-20/action_{W,S,A,D}_baseline30_124/baseline.mp4
```

本轮 causal 原始输出位于：

```text
outputs/2026-10-03-02/final_fixed_mix124_8step/{W,S,A,D}/cached.mp4
```

### 长视频指标

| Action | Original time (s) | Causal time (s) | Causal GPU peak | Causal CPU KV | Original mean RGB MAD | Causal mean RGB MAD | Original boundary MAD | Causal boundary MAD | Original horizontal flow | Causal horizontal flow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W | 441.5 | 450.2 | 39,927 MiB | 13,509 MiB | 4.32 | 4.60 | 4.97 | 6.29 | −1.111 | −0.624 |
| S | 444.8 | 451.9 | 39,927 MiB | 13,509 MiB | 4.13 | 4.62 | 4.70 | 7.76 | −1.084 | −0.096 |
| A | 454.2 | 438.8 | 39,948 MiB | 13,509 MiB | 4.52 | 7.10 | 5.14 | 13.68 | +1.077 | +0.143 |
| D | 450.4 | 383.4 | 39,948 MiB | 13,509 MiB | 4.46 | 5.39 | 5.39 | 9.47 | −1.602 | −0.311 |

`boundary MAD` 是 RGB 相邻帧在 chunk 边界帧 17、34、51、68、85、102、119 的均值；`MAD` 和 boundary MAD 都是连续性/运动描述，不是质量分数。causal rollout 仍然保持完整 124 帧和 cache lifecycle 正确，但 A/D 后半段出现明显 generated-history 漂移，边界 MAD 高于原始 H3。

A/D horizontal-flow action response：

```text
Original H3 A-D = +2.679
Causal fixed_mix A-D = +0.453
```

因此 causal branch 仍保留了 A>D 的正确符号，并且 W/S/A/D 会生成不同视频，但方向分离显著弱于原始 H3，不能写成“四方向完全保真”。W/S 的水平 flow 不作为前进/后退严格指标，因为本场景的前后运动可能投影成同号相机/深度运动；W/S 以并排视频作为 qualitative evidence。

### 最终可审阅产物

四行 4×2 对比网格（左：原始 H3 30 steps；右：scheduled-sampling causal 8 steps/chunk）：

```text
outputs/h3world_final_fixed_mix_action_grid_124.mp4
```

单 action 并排视频：

```text
outputs/h3world_final_W_original_vs_causal.mp4
outputs/h3world_final_S_original_vs_causal.mp4
outputs/h3world_final_A_original_vs_causal.mp4
outputs/h3world_final_D_original_vs_causal.mp4
```

统一指标和复现实验说明：

```text
outputs/2026-10-03-02/final_fixed_mix124_8step/final_action_summary.md
outputs/2026-10-03-02/final_fixed_mix124_8step/final_action_summary.json
outputs/2026-10-03-02/final_fixed_mix124_8step/action_flow.json
outputs/2026-10-03-02/final_fixed_mix124_8step/compact_continuity_metrics.json
```

本轮最终结论仍需保持克制：scheduled-history action residual 明显比无 adapter generated-history（39 帧 A-D≈0.015）更能保留动作敏感性，但长时生成分布偏移尚未解决；因此这验证了 `causal rollout + action interface` 的可行性，同时也验证了只靠 Stage1-style causal/teacher-forcing 或固定 history mixing 不能达到 SolarWM Stage2 的稳定少步长时 rollout。没有实现或声称 SGF/DMD。

补充的 pairwise action intervention 检查显示，causal 视频的 W-vs-S 平均 RGB 差异为 `29.92`，A-vs-D 为 `8.75`（`outputs/2026-10-03-02/final_fixed_mix124_8step/action_pair_differences.json`）。这只能说明动作干预产生了不同视频，不能替代方向正确性指标；A/D flow 分离和后半段视觉漂移仍应按上表报告。

## 2026-10-03：SolarWM 是否处理过 generated-history drift 的源码核对

已核对本地 `SolarWM/` 的 README、MiniMax-H3 Stage1/Stage2 配置和 SGF 实现。结论如下：

1. SolarWM 的公开训练路线明确把问题分成 Stage1 TF-AnyFlow 和 Stage2 SGF：Stage1 用 clean history 做 teacher-forced causal initialization；Stage2 则让 student 在自己的 autoregressive rollout 上训练，并用 frozen teacher + trainable critic 做 distribution matching。README 的原文是 `Stage2 performs DMD via self-gradient forcing (SGF), training the causal student on its own autoregressive rollout with a frozen teacher and a trainable critic`。
2. 因此 SolarWM 已经在方法层面处理了与本项目相同的 teacher-history / generated-history 分布差异和误差累积问题，但源码/README 没有专门公布 H3-World W/S/A/D action-collapse 的 ablation；不能说他们公开报告过同一个键盘动作失败案例。
3. SolarWM Stage2 不是简单把 clean latent 和 generated latent 做静态混合。`sgf_rollout.py` 中 student 先以 5-latent chunks 自回归生成，使用 detached raw KV cache；随后 `h3_sgf_replay` 用 rollout 的 noisy exit 和 generated clean target 重新 replay。`stage2_runtime.py` 再用 frozen Stage0.5 teacher 和 trainable critic 对 student output 做 score comparison，计算 SGF/DMD gradient，并更新 student；critic 每 5 次更新后才更新一次 student。
4. SolarWM 的 H3 action/control 语义与 H3-World 当前 W/S/A/D 不完全相同。SolarWM 的 `H3SGFInputs` 和 `H3SGFAttention` 主要接收每帧 `camera_viewmats`/`camera_K`，源码把 camera controls 视为 global immutable metadata；没有 H3-World 的 keyboard action rows/action_script 路径。因此 SolarWM Stage2 checkpoint 的成功不能直接证明 H3-World 的 W/S/A/D direction fidelity。
5. raw KV cache 和 sliding window 只保证历史上下文复用及可扩展的 causal window，不会自动纠正语义漂移；纠正漂移的是 Stage2 self-rollout + teacher/critic distribution matching。我们当前的 `history-mix` 是 detached static generated-history proxy，没有当前 adapter 的在线 self-rollout，也没有 SGF teacher/critic gradient，所以只能解释 39 帧 A-D 从 0.015 部分恢复，不能消除 124 帧后半段 drift。

对应源码：

```text
SolarWM/README.md
SolarWM/src/solarwm/backends/minimax_h3/stage1.py
SolarWM/src/solarwm/backends/minimax_h3/stage1_sampling.py
SolarWM/src/solarwm/backends/minimax_h3/sgf_rollout.py
SolarWM/src/solarwm/backends/minimax_h3/stage2_runtime.py
SolarWM/src/solarwm/backends/minimax_h3/sgf.py
SolarWM/configs/examples/minimax_h3/stage2-158f-lora384-w6-sp4.yaml
```

对本项目的直接启示是：如果继续做 SolarWM-inspired follow-up，核心应是在线 self-rollout 的 SGF-like 训练，并在 student、teacher、critic 三条路径中都保留 H3 action rows；继续堆叠普通 causal mask、KV cache、anchor 或静态 history mixing 不能等价替代 Stage2。
## 2026-10-03 17 时段：最小 action-aware online self-rollout teacher-replay

本轮按 `next_plan.md` 的收敛方案实现了一个小型 Stage2-inspired 诊断，目标是验证
SolarWM Stage2 的核心工程路径是否能迁移到 H3-World，而不是复现 SGF/DMD：

```text
student 自己 rollout 39 帧 / 3 causal chunks
→ 每个 chunk clean KV commit 时 detach 到 CPU
→ 用同一 generated history 建 frozen H3 teacher cache
→ student/teacher 在当前 chunk 上 replay
→ teacher velocity matching + 小 boundary loss
```

W/S/A/D action rows 在 student rollout、student commit、teacher commit、student replay 和
teacher replay 中都保留；teacher 只关闭 student 的 action residual，H3 的 packed action
condition 没有删除。新增脚本为
`H3-World/code/causal/train_online_selfrollout.py`。为支持当前 chunk 在读取 detached
KV 时保留梯度，`h3_cached.py` 增加 `allow_grad_read` 和 gradient-checkpoint 参数；DiT 的
causal control 允许 `allow_grad_read=True` 的 checkpoint；`CausalActionResidual` 增加运行时
`enabled` 开关，用于在同一 H3 实例上切换 student/frozen teacher。

### Smoke test

GPU 1 上完成 A 动作、39 帧、3 chunks、1 solver step、1 optimizer step：

- teacher-replay loss `0.0012899`，gradient norm `0.0121`，action adapter 参数发生更新；
- student forward 有梯度，KV commit 在 `no_grad` 中执行，缓存 history 校验通过；
- allocated GPU peak `42,981.9 MiB`，student/teacher CPU raw-KV 各约 `6.33 GiB`；
- 结果目录：`outputs/2026-10-03-16/online_selfrollout_smoke_A_ckpt2/`。

初版直接 GPU 反向在 4-step solver 上触发显存不足；随后启用 CPU activation checkpoint
offload，保留了 3 chunks 的真实 generated-history 训练路径。

### A/D online 训练

在 GPU 1 上从现有 fixed-mix action residual 初始化，A/D 交替做 4 个 optimizer steps，
每个 optimizer step 使用 4 solver steps/chunk：

| 项目 | 结果 |
|---|---:|
| latent/RGB frames | `12 / 39` |
| causal chunks | `3`（5+5+2 latent frames） |
| optimizer steps | `4`（A,D,A,D） |
| trainable parameters | `774,144` hidden action residual |
| replay loss | `0.001078, 0.001216, 0.001043, 0.001193` |
| wall time | `590.2 s` |
| allocated GPU peak | `41,344.8 MiB` |
| each CPU raw-KV cache | `6,799,104,000 bytes` |

adapter 和 trace 位于
`outputs/2026-10-03-17/online_selfrollout_AD_39_4x4_offload/`，其中
`training.json` 明确记录了 `history_source=online_student`、`history_detached=true`、
`student_forward_grad=true` 和 `commit_no_grad=true`。

同一时段的 `outputs/2026-10-03-17/online_selfrollout_AD_39_4x4/` 是未启用 CPU activation
offload 的失败尝试（显存不足），不作为结果；可复现实验以带 `_offload` 后缀的目录为准。

### 严格同条件回载对照

为了避免把 solver 或 anchor 差异误当作训练收益，fixed-mix 和 online adapter 都用：

```text
39 frames, seed=13, flow shift=2.22,
dynamic_last_frame_dual, 4 steps/chunk, generated history
```

| adapter | A horizontal flow | D horizontal flow | A−D |
|---|---:|---:|---:|
| fixed-mix 0.5 | `-0.0199` | `-0.3829` | **`+0.3630`** |
| online teacher replay | `-0.0210` | `-0.3703` | **`+0.3493`** |

online 的 A−D separation 比 fixed-mix 低 `0.0137`，属于基本持平，没有达到预设的
“online ≥ fixed-mix”成功标准。连续性 proxy 也基本持平：fixed A/D 的灰度 MAD 均值为
`2.963/2.895`，online 为 `2.933/2.893`；空间边缘差 fixed 为 `2.161/1.959`，online
为 `2.173/1.968`。这些是描述性指标，不是视频质量分数。

对应的 39 帧并排视频已放到输出根目录：

- `outputs/h3world_online_selfrollout_vs_fixedmix_A_39_dynamicdual.mp4`
- `outputs/h3world_online_selfrollout_vs_fixedmix_D_39_dynamicdual.mp4`

原始 JSON 指标：

- `outputs/2026-10-03-17/fixedmix_rollout_eval39_4step/AD_flow.json`
- `outputs/2026-10-03-17/online_rollout_eval39_dual/AD_flow.json`
- `outputs/2026-10-03-17/online_rollout_eval39_dual/continuity_comparison.json`

### 本轮结论

这一步证明了最小 online self-rollout 的**工程链路**可行：student 能在自身生成历史上继续
rollout，raw KV 可以 detach/复用，frozen H3 teacher 可以在相同 generated history 上
replay，action condition 仍然进入两条路径。它没有证明少步质量或 action fidelity 已经
改善：4 个 optimizer steps 的 hidden action residual teacher replay 没有超过已有 static
fixed-mix adapter，因此没有扩展到 124 帧，也没有把它称为 SolarWM Stage2。

当前更可信的面试结论是：

```text
causalization + persistent KV + action-aware online replay is runnable,
but a tiny teacher-replay update is insufficient to recover the 30-step
directional response. Stable long-horizon improvement needs a fuller
generated-history distribution-matching objective (SolarWM SGF/DMD-like
training), rather than another inference-only anchor trick.

### 为什么 online 结果仍然差

这次失败不是单一的 KV cache bug，而是目标、solver 和 H3 action 路径同时不匹配：

1. **4 steps 本身没有被蒸馏。** 当前 fixed-mix 在严格相同的 dynamic-dual、4-step 条件下
   A−D 只有 `+0.363`；同一类 30-step causal benchmark 的 A−D 约为 `+1.003`。online
   训练并没有实现 AnyFlow、SGF 或 DMD，因此不能把 4-step 当作已经具备 30-step score
   field 的少步模型。
2. **teacher replay loss 与 action fidelity 不一致。** student 只在 rollout 结束后的
   `sigma=0` 当前 chunk 上匹配 frozen teacher velocity，没有在 `sigma=1, 0.869, 0.689,
   0.425` 四个 solver 点上匹配 teacher map，也没有 A/D pair loss 或 action-margin loss。
   该目标更容易学共同的外观/运动，不能主动增大 A−D separation。
3. **student teacher 切换时关闭了 causal action residual。** 这使 teacher target 成为
   “没有 student causal residual 的 H3 replay”，而不是带有明确 action-direction target 的
   Stage0.5/Stage1 teacher。小 residual 的更新幅度也很小，4 个 optimizer steps 后各 block
   权重范数变化约为千分之几，说明这轮更接近 diagnostic smoke，而不是有效蒸馏。
4. **causal mask 改变了 H3 原有 action routing。** 当前训练使用
   `action_prefix_mode=own`、`action_feedback=False`。视频 query 可以读自己的 action
   row，但 action row 不能读回当前 video latent；H3 原始 directed action path 有这条
   feedback，released action LoRA 也是在 full-sequence bidirectional 条件下训练的。仅用
   hidden residual 很难补回这个被 causal cut 掉的反馈路径。
5. **dynamic latent dual anchor 不是 H3 原生 image anchor。** 生成的上一 chunk 尾帧经过
   latent patchify 后直接放入 image-like prefix，和 H3 的 RGB decode →
   `process_image=True` encode 语义不同；一旦前一 chunk 有漂移，anchor 会把漂移继续传给
   下一 chunk。

因此，“replay loss 下降”不能解释为“视频动作变好”。当前实验实际证明的是 cache/梯度/teacher
replay plumbing 可执行，而不是 causal model 已学会少步 action-conditioned score field。
若继续实验，优先级应是：先在固定 39 帧验证 `action_feedback=True` 的 action routing；再
把 teacher matching 放到每个 solver sigma，并加入 A/D pair or margin loss；最后才考虑
SGF-like critic。不要先扩展到 124 帧，也不要再用更多 anchor trick 掩盖这个目标错位。
```

## 2026-10-03 21--22 时段：action feedback、per-sigma replay 与 8-step 对齐实验

本轮目标是先解决当前视频“人物逐渐淡出、动作变弱”的问题，并把训练目标和最终
`8 steps/chunk` 推理对齐。所有 39 帧 A/D 对照都固定同一首帧、prompt、seed=13、初始
noise、`dynamic_last_frame_dual`、CPU raw-KV 和 H3 checkpoint；只替换 action 或 adapter。

### 1. Causal action routing ablation

新增的严格对照目录为：

```text
outputs/2026-10-03-21/action_feedback_ablation39/
```

在同一 `fixed_mix_0.5/action_adapter.pt`、4 steps/chunk 下：

| routing | A horizontal flow | D horizontal flow | A−D |
|---|---:|---:|---:|
| `action_prefix_mode=own`, feedback off | −0.0181 | −0.3834 | **0.3653** |
| `action_prefix_mode=causal`, `action_feedback=true` | −0.0496 | −0.4289 | **0.3793** |

恢复 action row 到当前 video latent 的 causal-safe feedback 后，A/D separation 有小幅提升，
但提升不足以解释全部画面退化。因此 action feedback 是必要条件，但不是完整解决方案。
原始结果位于 `own_nofb_flow.json` 和 `causal_fb_flow.json`。

### 2. Online teacher replay 改为匹配每个实际 solver sigma

`H3-World/code/causal/train_online_selfrollout.py` 现在支持：

```text
student 在自身 generated history 上 rollout
→ 每个 solver sigma 都用 frozen H3 teacher replay
→ local sigma loss 立即反向传播，solver state detach
→ clean sigma=0 replay 与 boundary loss
→ clean KV commit 后 detach 到 CPU
```

新增参数为 `--action-prefix-mode`、`--action-feedback`、`--sigma-replay-weight`。之前的
诊断只在每个 chunk 的 `sigma=0` 匹配，无法约束 4/8-step solver 使用的 noisy score field。
另外新增了可选的 `--causal-adapter --train-causal-adapter` 路径：student 可以训练已有的
tail QKV LoRA；teacher 在同一 DiT 中临时切换到冻结的 QKV snapshot，不复制第二份约 40 GiB
模型。QKV 训练结果同时保存为 `causal_adapter.pt`。

### 3. 与最终 8-step 推理对齐的 online 训练

输出目录：

```text
outputs/2026-10-03-21/online_sigma_feedback_39_8x4/
outputs/2026-10-03-21/online_sigma_feedback_39_8x4_eval/
```

配置为 39 RGB 帧、12 latent 帧、3 chunks、8 solver steps/chunk、4 optimizer steps，动作
顺序 A→D→A→D，`action_prefix_mode=causal`、`action_feedback=true`。结果：

| 项目 | 结果 |
|---|---:|
| replay loss（4 steps） | 0.005338, 0.005293, 0.005115, 0.005213 |
| online training wall time | 1,045.9 s |
| allocated GPU peak | 39,707.9 MiB |
| 每个 CPU raw-KV cache | 6,799,104,000 bytes（约 6.33 GiB） |
| A horizontal flow | −0.0131 |
| D horizontal flow | −0.8110 |
| A−D horizontal-flow separation | **0.7979** |

严格同条件的未再训练 `fixed_mix_0.5`、8 steps/chunk、feedback on 对照为 **0.7557**。
因此这轮 online per-sigma replay 在 39 帧短片上有小幅正向作用，但仍远低于原始 H3 teacher
的 A/D separation；它仍然是 Stage2-inspired diagnostic，不是 SGF/DMD。

### 4. 通用 tail QKV online smoke 与负结果

目录：

```text
outputs/2026-10-03-21/online_qkv_smoke39b/
outputs/2026-10-03-21/online_qkv_eval1_39_8step/
```

使用已有 tail16 causal QKV adapter，同时训练 3,440,640 个 QKV 参数和 774,144 个 action
residual 参数，1 optimizer step、4 solver steps/chunk：

- wall time `180.9 s`；
- allocated GPU peak `39,827.8 MiB`；
- student/teacher cache 各约 `6.33 GiB`；
- 训练链路和 `causal_adapter.pt` 保存均通过。

但只训练 1 step 后用 8-step 推理得到 A−D=`0.1674`，抽帧仍有明显淡出，因此不能把 smoke
当成质量改进。另一个冻结的旧 tail16 QKV adapter 与 fixed-mix action residual 叠加时，
39 帧 8-step 的 A−D 只有 `0.1596`。这两个结果说明通用 QKV adapter 需要更充分、与
generated-history 对齐的训练，不能直接作为最终 checkpoint。

### 5. 当前判断

8 steps/chunk 本身是本轮最明显的影响因素：同一 fixed-mix、feedback on 条件下，4-step
A−D=`0.3793`，8-step A−D=`0.7557`；8-step online per-sigma 训练进一步到 `0.7979`。
但是 39 帧视频抽帧仍可看到人物和场景随 chunk/history 逐渐淡出，说明核心瓶颈仍是
generated-history distribution shift 和少步 score field，而不是 KV cache lifecycle 或
显存不足。

目前不要把 39 帧 8-step online 结果扩展成 124 帧训练，也不要称为完成 SolarWM Stage2。
124 帧最终验收视频仍是：

```text
outputs/h3world_final_fixed_mix_action_grid_124.mp4
```

它对应的是之前的 `action_prefix_mode=own`、`action_feedback=false` fixed-mix checkpoint，
A/D 长时 separation=`0.453`；本轮 feedback/per-sigma 改动尚未重新生成 124 帧四方向网格。
下一步应先对 39 帧训练增加稳定的 action-pair/score matching 约束，并观察画面淡出是否
真正下降；只有短片画面稳定后，才重新制作 124 帧最终 demo。

## 2026-10-03 23 时段：N1 paired A/D teacher-delta objective

按照 `next_plan.md` v3，`code/causal/train_online_selfrollout.py` 已加入可选的
`--paired-action-loss`。在每个实际 solver sigma 上，A/D 两个 counterfactual 使用同一
generated latent、同一 detached student/teacher raw-KV history 和同一 sigma，分别计算：

```text
L_replay = 1/2 (MSE(vS_A,vT_A) + MSE(vS_D,vT_D))
L_dir    = 1 - cosine(vS_A-vS_D, vT_A-vT_D)
L_mag    = |norm(ΔS)-norm(ΔT)| / (norm(ΔT)+eps)
```

训练目标为 `L_replay + 0.1 L_dir + 0.1 L_mag`，继续保留 clean replay、boundary loss、
causal action prefix、action feedback、8 steps/chunk、dynamic dual latent anchor、CPU raw KV。
新增 `--checkpoint-every`，N2 可保存 `step_04/08/12/16/action_adapter.pt`。A/D 的
`prompt_embeds` 允许动作文本行不同；initial video noise 和 audio noise 要求逐元素一致，避免
把两个不同世界状态误当成 paired counterfactual。

N1 运行目录：

```text
outputs/2026-10-03-23/online_paired_sigma_39_8x4/
outputs/2026-10-03-23/online_paired_sigma_39_8x4_eval/
```

N1 配置是 4 optimizer steps、A→D→A→D、39 RGB frames、12 latent frames、3 chunks、8
solver steps/chunk、seed 13。训练 wall time `1,948.1 s`，allocated GPU peak `39,793.9 MiB`，
每份 student/teacher CPU KV 为 `6,799,104,000 bytes`（约 6.33 GiB）。训练中的均值如下：

| step | action | total | L_replay | L_dir | L_mag |
|---:|:---:|---:|---:|---:|---:|
| 1 | A | 0.066101 | 0.005902 | 0.366680 | 0.316511 |
| 2 | D | 0.075440 | 0.005681 | 0.394522 | 0.395598 |
| 3 | A | 0.060499 | 0.005746 | 0.346872 | 0.274806 |
| 4 | D | 0.071995 | 0.005648 | 0.388332 | 0.363436 |

N1 step-04 的 A/D 39-frame rollout 使用空闲 GPU1/GPU4 并行生成，采样各约 100 s，24 noisy
denoiser forwards、3 clean commits、CPU KV peak 约 6.33 GiB。动作 flow 结果：

| model | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| online per-sigma baseline | −0.0131 | −0.8110 | **0.7979** |
| N1 paired step-04 | −0.0063 | −0.7750 | **0.7687** |

A 仍未满足 `flow(A)>0`，A−D 也没有超过原 baseline，因此 N1 未通过 PASS gate。配对损失
本身已正常生效且训练没有 NaN/OOM，但 4 步对 action residual 的参数改变量很小，不能将
该负结果解释为 paired objective 已经无效。N2 的 16-step learning curve 已从统一
`fixed_mix_0.5/action_adapter.pt` 初始化启动，输出目录为：

```text
outputs/2026-10-03-23/online_paired_sigma_39_8x16/
```

在 N2 完成前不制作新的 124-frame 正式 grid；现有正式视频和 PASS gate 仍保持不变。

## 2026-10-04：N2 paired A/D 16-step learning curve

N2 目录：

```text
outputs/2026-10-03-23/online_paired_sigma_39_8x16/
outputs/2026-10-03-23/online_paired_sigma_39_8x16_eval/
```

从同一 `fixed_mix_0.5/action_adapter.pt` 开始，严格使用 Phase 0 协议和 A→D 交替共 16 个
optimizer steps；每 4 步保存 `step_04/08/12/16/action_adapter.pt`。训练 wall time
`6608.7 s`，allocated GPU peak `39,793.9 MiB`，student/teacher CPU KV 各约 `6.33 GiB`。
paired score 的数值目标确实下降：最后 A 步 `L_dir=0.2937, L_mag=0.1503`，最后 D 步
`L_dir=0.3019, L_mag=0.1999`；这说明优化器能匹配 teacher 的局部 A/D score delta。

但每个 checkpoint 用同一 39-frame、8-step、generated-history rollout 测得的动作流为：

| checkpoint | flow(A) | flow(D) | A−D | 39f gate |
|---|---:|---:|---:|---|
| step_04 | +0.0070 | −0.7898 | 0.7968 | A/D 方向通过，separation 不足 |
| step_08 | −0.0106 | −0.7832 | 0.7726 | 未通过 |
| step_12 | −0.0039 | −0.7894 | 0.7855 | 未通过 |
| step_16 | −0.0487 | −0.7898 | 0.7411 | 未通过 |

`learning_curve.md/json` 位于 `online_paired_sigma_39_8x16_eval/`。A/D 视频的 frame MAD 和
edge proxy 在四个 checkpoint 间没有出现能解释方向恢复的视觉改善。结论是：paired teacher
delta loss 在当前 774,144 参数 action residual 上可以降低局部 score-field loss，却不能把
这种局部几何传递到自身 generated-history 的 rollout；继续增加 paired steps 没有价值。

这满足“39 帧方向 gate 未通过”的分支判断：暂不做 124-frame 新 grid，也不做 SGF/DMD。下一
个最小实验改为 action-pathway adaptation，固定 causal mask、KV、anchor、chunk、solver 和
generated-history 协议，只解冻 H3 原 action LoRA 或 action-token refiner 的小参数集，并用
同样的 A/D teacher replay 重新测试；若仍不能达到 `flow(A)>0`、`flow(D)<0`、A−D>`1.0`，
再报告为 action representation 在 causal 拓扑下的限制。

## 2026-10-04：N3 action-token prefix residual smoke

为验证 action 文本行是否因 causal 拓扑变化而失配，新增了零初始化的
`CausalActionPrefixResidual`。它只向当前 chunk 的 action text prefix rows 注入由 9 维
action one-hot 产生的 hidden residual，参数量为 `387,072`；已有 video Q/K/V action
residual 冻结，causal mask、persistent CPU raw KV、dynamic dual anchor、8-step solver 和
generated-history 协议全部不变。

实验目录：

```text
outputs/2026-10-04-15/action_prefix_online_pair39_8x4/
outputs/2026-10-04-15/action_prefix_online_pair39_8x4_eval/
```

训练使用 A→D→A→D 四步 paired per-sigma teacher replay，学习率 `1e-5`，总耗时约
`1973 s`，allocated GPU peak 约 `39,925 MiB`，student/teacher CPU raw KV 各约 `6.33 GiB`。
4 步均正常完成并保存 `action_adapter.pt`、`action_prefix_adapter.pt` 及 step-04 checkpoint。
Rollout 的 paired loss 没有 NaN，最终 39-frame flow 为：

| 配置 | flow(A) | flow(D) | A−D | gate |
|---|---:|---:|---:|---|
| online per-sigma baseline | −0.0131 | −0.8110 | 0.7979 | 未通过 |
| prefix residual step-04 | **+0.0055** | **−0.7668** | 0.7724 | 方向通过，分离不足 |

prefix residual 让 A 恢复了正号，但没有达到 `A−D>1.0`，且分离度低于原始 online
baseline。因此 action text rows 不是唯一瓶颈，不能据此制作新的 124-frame grid，也不进入
SGF/DMD。下一步开始 H3 原始 action LoRA 的小范围低学习率适配：先只解冻 tail8 的原始
`qkv_proj/out_proj` LoRA，在 teacher replay 时交换回冻结的原始 LoRA，避免 teacher 随 student
漂移。

## 2026-10-04：H3 原始 LoRA 适配代码已接入

`train_online_selfrollout.py` 新增 `--train-h3-lora`、`--h3-lora-blocks` 和
`--h3-lora-lr`，可将指定 H3 block 的 released LoRA 矩阵作为 student 参数训练，同时在
teacher forward 中恢复原始矩阵。新增 `h3_lora_adapter.pt` 保存格式和 benchmark 的
`--h3-lora-adapter` 加载选项，避免把该实验误当成正式 H3 checkpoint。代码通过 py_compile，
`tests/test_pretrained_lora.py tests/test_h3_cached.py` 共 7 项通过。

当前仍未满足 39-frame PASS gate：

```text
flow(A) > 0, flow(D) < 0, A−D > 1.0
```

所以新的 124-frame 视频、action switching grid 和 SGF/DMD 均继续冻结。prefix smoke 的原始
视频和指标保留在上述时间目录中，正式 124-frame deliverable 仍为
`outputs/h3world_final_fixed_mix_action_grid_124.mp4`。

## 2026-10-04：原始 H3 LoRA tail8 单步 smoke

为检查 causal action representation 是否需要直接适配 H3 已发布的 action LoRA，新增了
一个单步实验：只训练 blocks `42--49` 的 `qkv_proj/out_proj` LoRA（共 `10,092,544`
参数，学习率 `1e-6`），已有 causal video action residual 冻结；teacher forward 每次恢复
原始 H3 LoRA 矩阵。代码路径和 adapter 文件为：

```text
outputs/2026-10-04-17/h3_lora_tail8_online_pair39_8x1/
outputs/2026-10-04-17/h3_lora_tail8_online_pair39_8x1_eval/
```

单步训练耗时约 `693 s`，allocated GPU peak 约 `41,576 MiB`，没有 NaN/OOM，成功写出
16 个 tail8 LoRA module 的 `h3_lora_adapter.pt`。但同一 39-frame rollout 的 flow 为：

| 配置 | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| online per-sigma baseline | −0.0131 | −0.8110 | 0.7979 |
| tail8 H3 LoRA 1-step | −0.0134 | −0.7956 | 0.7822 |

单步更新没有改善 A 的正向符号，分离度也略低于 baseline；抽帧仍显示约第 20 帧后人物
雾化和 ghosting。因此暂不投入昂贵的 4-step tail8 训练，先把该结果作为 action-pathway
负诊断保存。当前证据更支持：问题主要来自 causal generated-history 的 score-field/
distribution shift，而不是仅靠解冻 tail LoRA 就能修复的 action token 映射。

本轮新增的 `h3_online_lora_v1` adapter 只用于实验回载，benchmark 通过
`--h3-lora-adapter` 加载，不能称作新的 H3 checkpoint。39-frame gate、124-frame grid、
action switching 和 SGF/DMD 仍未启动。

## 2026-10-04：action-prefix `all` routing 上界

为区分 action-row routing 与 generated-history drift，固定 fixed-mix action adapter、8
steps/chunk、dynamic dual anchor、CPU KV 和 generated history，只把
`action_prefix_mode` 从 `causal` 改为 `all`，并行生成 A/D 39 帧。输出位于：

```text
outputs/2026-10-04-18/action_mode_all_fixedmix39/
```

结果为：

| action prefix mode | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| causal（fixed-mix 参考） | 约 +0.02 | 约 −0.74 | 约 0.756 |
| all（本次上界诊断） | +0.0513 | −0.1508 | 0.2021 |

`all` 并没有恢复动作，反而让 D 的响应接近消失，说明把未来 action rows 暴露给当前视频
查询会产生跨时间控制污染；问题不是简单的“causal mask 少看了未来 action”。因此不采用
`all` 作为模型方案，继续固定 `action_prefix_mode=causal`。当前证据将重点收敛到
generated-history distribution shift 与少步 score field，而不是继续放宽 action mask。

## 2026-10-04：generated chunk endpoint matching smoke

为直接测试生成历史是否可以通过 teacher latent anchor 稳定，给 online paired replay 增加了
可选的 `--latent-target-weight`。它在每个 chunk 的最后一个 solver step，把 student 的
最终 latent 与同 action、同 initial noise 的原始 H3 30-step `baseline_latents.pt` 做 endpoint
MSE；不改变 causal mask、KV cache、anchor、solver 或 action routing。新增的日志字段为
`latent_target_loss_mean`。

单步、权重 `0.1` 的 smoke 输出为：

```text
outputs/2026-10-04-18/online_target_pair39_8x1_retry/
outputs/2026-10-04-18/online_target_pair39_8x1_retry_eval/
```

训练正常完成，`latent_target_loss_mean=0.2469`，无 NaN/OOM；但 39-frame rollout 为：

| 配置 | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| online per-sigma baseline | −0.0131 | −0.8110 | 0.7979 |
| endpoint target 1-step | −0.0110 | −0.7569 | 0.7459 |

动作分离没有改善，短片后段 ghosting 也没有消失，因此不继续扫描 endpoint 权重。结合
paired score、prefix residual、tail8 H3 LoRA 和 `action_prefix_mode=all` 的结果，当前廉价
action-pathway 修复均未通过 gate；问题更像 generated-history distribution shift 与少步
score field，而不是单个 action token 或未来 action-row 可见性。

## 2026-10-04：最小 Stage2-inspired two-pass replay

为直接测试 generated-state 分布匹配，`train_online_selfrollout.py` 新增了
`--final-replay-weight` / `--final-replay-sigma`。每个 optimizer step 先完成 detached 的
3-chunk student rollout，再重建前两 chunk 的 student/teacher raw KV，并在最终生成 chunk
的自身 state 上用固定 sigma 做一次 student/teacher replay。这是 SolarWM Stage2 的最小
诊断版本，没有 critic、DMD、SGF KL gradient 或第二个 score role。

单步 smoke（weight=`0.1`、sigma=`0.6`）目录：

```text
outputs/2026-10-04-19/stage2_minimal_final_replay39_8x1_retry2/
outputs/2026-10-04-19/stage2_minimal_final_replay39_8x1_retry2_eval/
```

训练正常完成，`final_replay_loss_mean=0.000637`，耗时约 `456 s`；39-frame rollout 为：

| 配置 | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| online per-sigma baseline | −0.0131 | −0.8110 | 0.7979 |
| minimal two-pass replay | −0.0008 | −0.7551 | 0.7543 |

动作分离没有改善，视频后段 ghosting 仍然存在。因此这个两阶段 replay 不能替代
SolarWM Stage2 的 critic/score-role 分布匹配，不能称为 Stage2 成功，也不继续做无意义的
权重扫描。当前实验链已经覆盖 action prefix、原始 H3 tail LoRA、未来 action-row 上界、
endpoint latent matching 和最小 two-pass replay；39-frame PASS gate 仍未通过。

## 2026-10-04：当前实验归档

为便于面试演示和复核，最新几轮结果已汇总到：

```text
outputs/2026-10-04-19/final_report/REPORT.md
outputs/2026-10-04-19/final_report/summary.json
outputs/2026-10-04-19/final_report/action_ablation_contact_sheet.jpg
```

报告统一列出 online baseline、prefix residual、tail8 H3 LoRA、`action_prefix_mode=all`、
endpoint target 和 minimal two-pass replay 的 A/D flow。当前没有任何新 checkpoint 达到
`flow(A)>0`、`flow(D)<0`、A−D>`1.0`，因此唯一正式 124-frame 对照仍是
`outputs/h3world_final_fixed_mix_action_grid_124.mp4`。最终结论应把 Stage1 causal/KV
feasibility 与 Stage2 generated-history quality limitation 分开陈述。

## 2026-10-06：anchor 作用范围、full-teacher action geometry 与 visual/action 折中

本轮继续围绕“人物分解”和 A/D action collapse 做了严格的 39-frame 诊断。所有结果均固定
12 latent frames、3 chunks、chunk size 5、8 steps/chunk、flow shift 2.22、seed 13、
generated history、CPU raw KV、`action_prefix_mode=causal` 和 `action_feedback=true`；没有
扩展到 124 帧。

首先给 `benchmark.py` 增加了 `--causal-adapter-scope {all,commit,last_step_commit}`。目的是
让训练得到的 visual tail16 QKV adapter 只负责 clean KV commit，而不改变当前 chunk 的
高噪声 action score field。用原始 W-trained tail16 adapter 做 A/D 对照时：

| scope | flow(A) | flow(D) | A−D | 观察 |
|---|---:|---:|---:|---|
| `commit` | −0.0266 | −0.0494 | 0.0228 | 人物仍会后段分解，动作基本消失 |
| `last_step_commit` | −0.0260 | −0.0616 | 0.0356 | 与 `commit` 相同，不能恢复动作 |

随后把同一个 scope 传入 `train_online_selfrollout.py`，分别做 4-step paired replay。两个
训练都正常结束，没有 NaN/OOM；`commit` 的第 4 步 `L_dir=0.1415`，`last_step_commit` 为
`0.1317`，但对应 rollout 的 A/D separation 只有约 0.022/0.012。说明训练/推理 scope
一致本身不能解决 causal action geometry。

一个更关键的修正是：此前的 paired delta loss 比较的是“同一个 causal mask 下的 teacher”，
而这个 teacher 的 A/D delta 已经随 causal 拓扑塌缩，不能作为原始 H3 action geometry 的
监督。因此 `train_online_selfrollout.py` 新增了 `--paired-full-teacher` 和
`full_teacher_forward`：在同一 generated state 上关闭 causal controller，使用原始 H3
双向 attention 计算 A/D teacher velocity，再对齐 student 的 action delta。1-step smoke
的目标确实不同（`L_dir=0.9566`，而 causal-teacher 约 0.14），但 rollout 仍只有
`flow(A)=0.0438`、`flow(D)=-0.5871`、A−D=`0.6309`。把高幅 action residual 与 tail16
QKV 联合更新 1 step 也没有改变这一结果。该链路证明了监督方向正确，但 1 step 不足以把
full-teacher geometry 迁移到 causal rollout，不能称为 Stage2 成功。

本轮最终定位了人物分解的主要工程原因：tail16 visual adapter 是按
`dynamic_last_frame_rgb_dual` 训练的，而最近诊断使用了便宜的 `dynamic_last_frame_dual`
latent anchor。换回训练一致的 RGB dual 后，人物和停车场结构在 39 帧内明显稳定：

| anchor / adapter | flow(A) | flow(D) | A−D | 视觉观察 |
|---|---:|---:|---:|---|
| latent dual + fixed visual | −0.698 | −0.691 | −0.008 | 人物后段透明/分解 |
| RGB dual + fixed visual | −0.881 | −0.756 | −0.124 | 人物全程基本完整，但动作同向 |
| RGB dual + action residual ×64 | +0.047 | −0.613 | 0.660 | A 方向恢复，D 开始 ghosting |

对 action residual 做了 8/16/32/48/64 倍幅度扫描。单一 global gain 不能同时保持 A/D 的
稳定性：A 需要很大幅度才翻到正号，而 D 在相同幅度下会分解。作为机制诊断，构造了一个
单文件的 per-action gain 原型（A 列 ×64、D 列 ×8），仍使用 RGB dual 和同一 checkpoint
加载路径。它的 39-frame 结果为：

```text
flow(A) = +0.0474
flow(D) = -0.7812
A-D     =  0.8286
```

抽帧中 A、D 两条人物都保持完整，动作方向也正确；但 separation 仍低于严格 gate
`A-D>1.0`，而且 per-action gain 目前只是 inference ablation，不是训练得到的最终模型。
完整指标和 contact sheet 位于：

```text
H3-World/outputs/2026-10-06-06/action_gain_sweep/RESULTS.md
H3-World/outputs/2026-10-06-06/action_gain_sweep/summary.json
H3-World/outputs/2026-10-06-06/action_gain_sweep/rgb_gain_A64_D8/eval/contact_sheet.jpg
```

这一目录下的 MP4 已用 PyAV 验证为 H.264、YUV420P、832×480、39 帧，可正常解码。训练一致
RGB dual 的代码参数也已经保留在 `benchmark.py`，并不会触发 ModelScope 下载。

当前结论更新为：RGB dual 修复了主要的视觉崩坏，per-action gain 证明 action geometry
可以被重新激活，但尚未达到统一 checkpoint 的 39-frame PASS gate。因此不制作新的
124-frame action grid，也不把该 inference gain ablation 称为 SolarWM Stage2。下一步若
继续，应该把 per-action gain 作为可训练参数并在 full-attention teacher 的 counterfactual
目标下优化；在此之前继续扫 solver、anchor 或重复 causal-teacher paired loss 没有归因价值。

## 2026-10-06：per-action gain 已接入可训练路径

为避免把 A×64/D×8 的手工缩放误当成模型能力，`CausalActionResidual` 现在包含一个可选的
FP32 `action_gain[9]` 参数。旧版 `h3_causal_action_residual_v1` checkpoint 没有这个字段时
自动使用全 1，因此所有已有 adapter 保持兼容。`train_online_selfrollout.py` 新增：

```text
--action-gain-init A=64,D=8
--action-gain-lr 1e-2
--anchor-mode {latent,rgb}
--paired-full-teacher
```

`--anchor-mode rgb` 会在每个 generated chunk 边界执行真实的 RGB decode/re-encode，和
tail16 visual adapter 的训练协议一致。用 RGB anchor、full-attention counterfactual teacher、
A=64/D=8 初始化做了 1-step smoke：训练约 10.5 分钟，峰值约 40.1 GiB，无 NaN/OOM；gain
参数从 `[A=64,D=8]` 只发生了很小变化（A≈63.98，D≈8.01），这说明 optimizer 和保存/回载
路径已经生效，但 1 step 不足以改变 rollout。记录位于：

```text
H3-World/outputs/2026-10-06-07/gain_lr_smoke_rgb_8x1/training.json
H3-World/outputs/2026-10-06-07/gain_lr_smoke_rgb_8x1/step_01/action_adapter.pt
```

相关代码通过 `py_compile`，已有 `tests/test_pretrained_lora.py tests/test_h3_cached.py`
共 7 项测试通过。当前最佳可播放短片仍是 RGB dual + A64/D8 inference ablation（A−D≈0.829）；
gain 训练需要继续做 4-step learning curve 才能判断是否能超过 1.0，之前不生成新的 124-frame
grid。

## 2026-10-06：开始 per-action gain 4-step learning curve

在完成 1-step RGB-anchor/full-attention-teacher smoke 后，启动了严格固定协议的 4-step 训练：

```text
output: outputs/2026-10-06-08/trainable_gain_full_teacher_rgb_8x4/
GPU: 0（GPU 2/3/5/7 上的其它 VLLM 进程未触碰）
39 RGB frames / 12 latent frames / 3 chunks
chunk=5 / history=5 / 8 solver steps/chunk
flow shift=2.22 / seed=13
generated history / CPU raw KV / action_prefix_mode=causal
action_feedback=true / RGB dual anchor
full-attention H3 A/D counterfactual teacher
trainable action gain init: A=64, D=8
action-gain-lr=1e-2 / base lr=5e-5
checkpoint every step
```

该实验只改变 action gain 的训练状态，不再改变 causal adapter、anchor、solver、chunk 或
KV 配置。第一次启动在 `step_01` 写出后发现旧前向把 FP32 gain 提前转换成 BF16（例如
8.0092 实际变成 8.0），因此已中断并标记为实现诊断，不把它当作完整曲线。现在
`CausalActionResidual.project_action` 对小型 action projection 使用 FP32，只有完成的残差
才转回 DiT 的 BF16；旧 checkpoint 仍按 legacy projection 路径兼容加载。修正后的 4-step
曲线将放在新的独立目录，仍会逐步保存 `step_01` 至 `step_04` 并统一生成 A/D 39 帧，计算
horizontal flow、A−D、boundary/frame MAD 与 contact sheet。只有同时满足 `flow(A)>0`、
`flow(D)<0`、A−D>`1.0` 且人物完整，才会进入 124-frame 阶段。若曲线仍停在约 0.8，则
接受当前 action geometry 受 causal/generated-history 分布限制的结论，不把 gain ablation
或普通 replay 误称为 SolarWM Stage2。

修正前实验目录：

```text
outputs/2026-10-06-08/trainable_gain_full_teacher_rgb_8x4/
```

其中 `step_01` 可作为“旧 BF16 gain path”诊断证据，训练状态为 interrupted；它不会被提升为
最终模型。修正同时新增了 gain 精度/旧 checkpoint 兼容测试，相关测试总数为 9 项。

## 2026-10-06：tail4 action-QKV paired alignment 4-update curve 已完成

在 RGB-consistent anchor、generated history、persistent CPU raw KV、8 steps/chunk、3 chunks
和 seed=13 全部冻结的条件下，完成了 `stage2_lite_dmd.py --paired-delta --paired-only`
的 4 个 optimizer update。只训练 tail4 action-conditioned Q/K/V refiner；每轮都保存了
`student_action_adapter.pt`，并用同一个 checkpoint 做 39 帧 A/D 自回归评估。训练使用
同一 generated state 的 full-attention H3 A/D counterfactual delta，记录四个 sigma 和两个
target chunks 的 `L_dir`、`L_mag`、student/teacher delta norm。

结果汇总如下。flow 是固定中心裁剪上的水平 Farneback flow，只是动作响应代理；严格 gate 为
`flow(A)>0`、`flow(D)<0`、`A-D>1.0`，并要求人物/车库结构保持稳定。

| update | flow(A) | flow(D) | A-D | mean paired direction cosine | mean norm ratio | gate |
|---:|---:|---:|---:|---:|---:|:---|
| 1 | -0.8563 | -0.7790 | -0.0774 | 0.3901 | 1.018 | FAIL |
| 2 | -0.8886 | -0.7536 | -0.1350 | 0.3474 | 0.973 | FAIL |
| 3 | -0.8703 | -0.7785 | -0.0918 | 0.3448 | 1.019 | FAIL |
| 4 | -0.8506 | -0.7684 | -0.0822 | 0.4242 | 1.045 | FAIL |

RGB dual anchor 继续有效：四轮视频中的人物和停车场结构都保持到第 38 帧，没有此前 latent-only
anchor 的透明/分解。可是 A/D 仍然表现为几乎相同的共同场景运动，paired score-field 的内部
对齐（norm ratio 接近 1、cosine 约 0.34–0.42）没有传递到自由 generated-history 的图像空间
动作方向。也就是说，这一轮排除了“tail4 paired update 不够多”这一简单解释，进一步支持
`action rows/feedback -> causal video-token routing` 与原始 H3 拓扑不一致的判断。

完整逐视频指标、训练 loss 曲线和 contact sheet：

```text
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_ALIGNMENT_LEARNING_CURVE.md
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_ALIGNMENT_LEARNING_CURVE.json
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/action_alignment_learning_curve_contact_sheet.jpg
```

当前决策：不再继续增加 tail4 paired updates，不再做新的 gain/anchor/solver sweep，也不扩展
到新的 124 帧 action grid。正式 124 帧主 demo 仍是旧的 fixed-mix 结果；本轮作为“内部 score
alignment 改善但自由 action geometry 未恢复”的严谨负结果保留。若要继续，下一次有归因价值的
改动应是直接检查/重构 causal action routing（例如保留原始 action-to-token 路径或显式 action
token refiner），而不是继续扩大同一 QKV loss。

## 2026-10-06：action-routing probe 证明 feedback edge 存在但贡献很小

新增 `H3-World/code/causal/probe_action_routing.py`，不训练模型，只在同一 generated A history、
同一 chunk 1 noisy state、同一 RGB dual anchor、sigma=0.6 上比较 action prefix visibility 与
`action_feedback`。A/D score delta norm 为：

| variant | A/D delta norm | 相对 causal/no-feedback |
|---|---:|---:|
| own + feedback off | 5.502 | 0.748x |
| causal + feedback off | 7.352 | 1.000x |
| causal + feedback on | 7.642 | 1.039x |
| all + feedback on | 8.338 | 1.134x |

`causal_fb1` 明确不同于 `causal_fb0`，说明 action-row 到 current-video 的反馈边确实生效；
因此不能把当前失败简单归因于 causal mask 完全切断 action rows。另一方面，约 643 的总 velocity
norm 对应的动作 delta 只有约 1.2%，与 teacher delta audit 的弱信号一致。当前更合理的解释是
动作 representation/score geometry 在 generated-history causal rollout 下被压弱或旋转，而不是
缺少一条边。

详细记录：

```text
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_ROUTING_PROBE.md
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/action_routing_probe.json
```

下一步保持 mask、KV、RGB anchor 和 solver 不变，先测原始 H3 action LoRA 的 action-row hidden
与 video output sensitivity；只有确认单 chunk 有正确 signal、但多 chunk 后 signal 消失时，才
进入 generated-history distribution matching。当前不实现未经证实的 bypass，也不生成新的
124-frame grid。

## 2026-10-06：冻结 causal 与原始 H3 的 action geometry 直接对照

在同一个 generated A history（chunk 0）、同一个 chunk 1 noisy latent、RGB dual anchor、prompt、
audio noise 和 sigma=0.6 上，比较冻结 visual causal adapter（causal prefix + action feedback）
与原始双向 H3 teacher 的 A/D counterfactual velocity。未安装 student action-QKV adapter，以排除
paired refiner 的影响。

| quantity | frozen causal | original H3 teacher |
|---|---:|---:|
| A/D delta norm | 7.642 | 8.704 |
| causal / teacher norm ratio | 0.878 | — |
| A velocity norm | 642.857 | 599.640 |
| D velocity norm | 643.615 | 598.936 |
| delta cosine | **-0.015** | — |

动作 delta 的幅度接近 teacher，但方向几乎正交。因此当前失败不是 action signal 完全消失，也
不是简单增大 gain 可以解决；causal chunk attention 加 generated-history 改变了 action-conditioned
score field 的方向。这与 routing probe 的结论一致：feedback 边存在，但它没有把 action delta
旋转回原始 H3 的 image-space geometry。

详细报告：

```text
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_GEOMETRY_PROBE.md
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/action_geometry_probe.json
```

当前不再实现未经证实的 action bypass，也不继续 gain/paired-QKV sweep。若继续实验，唯一有归因
价值的方向是 action representation/score-field adaptation，且必须先在相同 39-frame gate 上
验证，再考虑任何长视频或 Stage2 扩展。


## 2026-10-06：action-prefix representation 1-step smoke

为区分“video-token QKV 不足”和“action row 表示本身需要重映射”，冻结现有 video action residual
与 RGB-consistent visual adapter，只训练 tail8 的 `CausalActionPrefixResidual`，目标仍是 full-attention
H3 的 A/D counterfactual delta。配置保持 39 帧、3 chunks、8 steps/chunk、RGB dual、generated
history、CPU raw KV、seed=13。

训练约 504.5 秒，峰值 allocated GPU 约 40.1 GiB，387,072 个 prefix 参数，无 OOM/NaN。paired
loss=1.0807、direction loss=0.9756、magnitude loss=0.9608。用 step_01 checkpoint 做同协议 A/D
rollout：

| variant | flow(A) | flow(D) | A-D | gate |
|---|---:|---:|---:|:---|
| tail8 action-prefix, update 1 | -1.1091 | -1.4603 | +0.3513 | FAIL |

抽帧中人物和车库结构保持到第 38 帧；A/D 有轻微区分，但两个 flow 仍为负，separation 远低于
1.0。该结果说明 action-prefix 表示适配可能比 tail4 QKV 更有方向性，但一个 update 尚不足以
恢复 image-space geometry，也不支持直接扩大训练预算。

报告和视频：

```text
H3-World/outputs/2026-10-06-08/action_prefix_align_full_teacher_rgb_39_8step_1update/ACTION_PREFIX_ALIGNMENT_REPORT.md
H3-World/outputs/2026-10-06-08/action_prefix_align_full_teacher_rgb_39_8step_1update/action_flow.json
H3-World/outputs/2026-10-06-08/action_prefix_align_full_teacher_rgb_39_8step_1update/contact_sheet.jpg
```

当前不把该 checkpoint 作为主方案，不生成 124-frame grid；继续 prefix 训练前需要新的训练
目标或多状态监督，而不是简单把 optimizer steps 增大。


## 2026-10-06：原始 H3 action LoRA tail8 1-step smoke

最后做了一个更接近原始 H3 action pathway 的适配：冻结 visual RGB causal adapter 和 video action
residual，只微调 released H3 attention LoRA 的 tail8 QKV/out matrices，目标仍为 full-attention
H3 A/D counterfactual delta。训练 526.8 秒，10,092,544 个 LoRA 参数，峰值约 40.14 GiB GPU、
CPU raw KV 6.33 GiB，无 OOM/NaN。

39-frame rollout：

| variant | flow(A) | flow(D) | A-D | gate |
|---|---:|---:|---:|:---|
| H3 action LoRA tail8, update 1 | -1.1249 | -1.4151 | +0.2902 | FAIL |

A/D 仍同向负 flow，且 separation 略低于 action-prefix smoke 的 0.3513。至此 tail4 action-QKV、
action-prefix hidden residual、原始 H3 action LoRA 三类 action-path adaptation 都没有在短片上
恢复 `A>0,D<0,A-D>1.0`。这轮作为最终结构性负结果保留，不再扩 optimizer steps 或生成
124-frame grid。

报告：

```text
H3-World/outputs/2026-10-06-08/action_h3_lora_align_full_teacher_rgb_39_8step_1update/ACTION_H3_LORA_ALIGNMENT_REPORT.md
H3-World/outputs/2026-10-06-08/action_h3_lora_align_full_teacher_rgb_39_8step_1update/action_flow.json
```

## 2026-10-06：action 实验收敛与最终诊断报告

已把 RGB Stage2-lite、tail4 action-QKV 4-update、tail8 action-prefix 1-update、released H3
action-LoRA tail8 1-update 统一收集到：

```text
H3-World/outputs/2026-10-06-08/FINAL_ACTION_DIAGNOSTIC.md
```

共同结论是：RGB-consistent anchor 已解决早期人物分解，8 个新增 39-frame 视频都能以
H.264/YUV420P 正常解码并保持场景结构；但所有 action-path adaptation 都未同时满足
`flow(A)>0`、`flow(D)<0`、`A-D>1.0`。冻结 geometry probe 显示 causal A/D delta 幅度接近
teacher、方向却几乎正交（cosine=-0.015），routing probe 显示 feedback 边存在但影响很小。

因此 Stage1/Stage2-lite 的最终边界已经清楚：causal/KV 工程可行，generated-history 下的
H3 action geometry 尚未恢复。后续若要突破，需要多状态/多 seed action supervision 或真正的
SolarWM Stage2 rollout-distribution matching；不再继续单场景单 state 的 LoRA/gain/anchor
sweep，也不生成新的正式 124-frame action grid。

## 2026-10-06：面试题提交包整理完成

已在 `/home/qma/work/GWM/submission`（当前源项目真实路径为 `/home/lpeng/code/mq_PubDataset/GWM/submission`）整理干净提交包，包含：

- `INTERVIEW_ANSWER.md`：SolarWM Stage0.5/Stage1/Stage2、KV cache、H3-World 流程和迁移结论；
- `EXPERIMENT_REPORT.md`：39/124 帧协议、视觉稳定性、Stage2-lite 和 A/D geometry 结果；
- `REPRODUCE.md`：外部权重、DiffSynth patch、因果 benchmark 和视频验证命令；
- `code/causal`、`code/abot` 和 DiffSynth patches；
- 小型 RGB visual、Stage2-lite 和 action diagnostic adapters；
- H.264/YUV420P 可解码的原始-vs-causal、视觉修复、Stage2-lite 和 action geometry 视频；
- `breakthrough/01` 至 `breakthrough/05`：每个关键突破的问题、方案、证据视频和限制。

提交包没有复制 33B 基础权重、H3-World 基础 LoRA、数据集、`.cache`、`__pycache__`、latent 或 conditioning 中间文件。23 个收录视频已用 PyAV 验证为完整可解码的 H.264/YUV420P、24 fps；25 个 Python 源文件通过语法检查。当前包约 87 MB。

提交包最终结论保持诚实：causal chunk rollout、persistent KV 和长时视觉稳定性已验证；generated-history 下的原始 H3 A/D action geometry 仍未通过 `flow(A)>0, flow(D)<0, A-D>1.0` gate，因此不声称已经完整保留 H3-World action control。

## 2026-10-06：新增会议展示包

已在 `/home/qma/work/GWM/submission/meeting` 建立现场展示材料：

- `annotated/h3world_final_action_grid_124_timed.mp4`：W/S/A/D 四行总览，左侧原始 H3 30 steps，右侧 causal 8 steps/chunk，并标注每个动作的 recorded end-to-end 时间；
- `annotated/h3world_final_{W,S,A,D}_original_vs_causal_timed.mp4`：带步数、总耗时、124 帧/5.17 秒和 causal forward/commit 口径的并排视频；
- `METRICS.md`、`METRICS.csv`：逐动作耗时、首块/平均块延迟、峰值显存、CPU raw KV、相邻帧 MAD、块边界 MAD 和 Farneback 水平光流；
- `FAIRNESS.md`：同首帧、prompt、动作、seed、初始 noise、分辨率和帧数的公平性说明，以及 30 full-horizon steps 与 8 steps/chunk 的计数口径；
- `SLIDES.md`、`MEETING_SCRIPT.md`：8 页展示提纲和约 5 分钟讲稿。

会议材料明确注明当前耗时是每个动作一次 recorded run，不是 warmup 后多次均值；也明确说明 formal fixed-mix causal grid 的 64 noisy forwards 高于原始 30 forwards，因此当前结论是 causal feasibility、历史复用和视觉修复，而不是端到端加速或完整 action preservation。
