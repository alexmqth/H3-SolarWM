# Next Plan v3：在 generated history 下恢复 H3-World 的 action geometry

更新时间：2026-10-06


## Latest result: endpoint latent supervision is also a negative smoke

A one-update tail4 action-QKV experiment added a direct target from the original H3 30-step A/D
`baseline_latents.pt` at generated chunk endpoints, on top of the full-teacher paired score-delta
loss. It used the frozen RGB visual adapter, RGB dual anchor, generated history, CPU raw KV, 8
steps/chunk, target chunks 1 and 2, and four solver sigmas. The rollout gave:

```text
flow(A) = -0.8041
flow(D) = -0.7047
A-D     = -0.0994
latent delta cosine = 0.0563
```

The previous endpoint-free latent delta cosine was about `0.0160`, so the target provides a small
internal rotation but does not recover the image-space action direction. Do not increase this
experiment's update count. The checkpoint and report are under
`outputs/2026-10-06-16/action_qkv_latent_endpoint_pair_rgb_39_8step_1update/`, and the playable
diagnostic is `outputs/stage2_action_qkv_latent_endpoint_AD_39.mp4`.

The action gate is still the decision boundary, not an internal cosine:

```text
flow(A)>0, flow(D)<0, A-D>1.0, person/garage stable through frame 38
```

The remaining defensible research path is multi-state/multi-seed training on the student's own
rollout distribution, or a causal action-token/attention topology change. Keep the RGB visual
adapter frozen and stop single-state action adapter, gain, anchor, endpoint-weight, and routing
sweeps. Do not create another formal 124-frame action grid until the short gate passes.

The first endpoint smoke had a mixed-state target (D endpoint from an independent D rollout applied
to an A-generated counterfactual). That protocol is retained only as a negative diagnostic. The
corrected own-history endpoint run applies A targets on A history and D targets on D history; it still
gives `flow(A)=-0.8086`, `flow(D)=-0.6893`, `A-D=-0.1194`, and latent delta cosine `0.0655`. Its
report is under `outputs/2026-10-06-16/action_qkv_own_endpoint_pair_rgb_39_8step_1update/` and the
playable comparison is `outputs/stage2_action_qkv_own_endpoint_AD_39.mp4`. The corrected protocol
removes the state-mismatch confound, but it does not justify more single-state updates.


## Latest execution update: routing and still controls are negative diagnostics

The RGB visual adapter remains frozen as the visual-stability baseline. A routing upper bound was
run with `action_prefix_mode=all` and the same generated-history / RGB-dual / CPU-KV / 8-step
protocol. It produced `flow(A)=-1.1535`, `flow(D)=-1.4733`, `A-D=0.3198`, essentially the same
as causal prefix visibility. A same-seed `STILL` rollout produced `flow=-0.7129`; subtracting it
left A=`-0.4406` and D=`-0.7604`, so common scene drift is not the explanation either.

This closes two low-level hypotheses. Do not spend more runs on `action_prefix_mode`, routing edge,
flow-shift, anchor, gain, or solver sweeps. The remaining action failure is a representation/topology
mismatch between the original bidirectional H3 action-conditioned score field and the generated-state
causal score field. The experiment and playable A/D comparison are in:

```text
outputs/2026-10-06-16/routing_all_rgb_39_8step/
outputs/stage2_routing_all_rgb_AD_39.mp4
```

If more training is required, the next defensible experiment is a multi-state action objective that
uses original H3 latent endpoints or image-space/counterfactual transition targets on several seeds,
while keeping the RGB visual adapter and inference protocol fixed. A single-state score-delta/QKV
loss should not be expanded again. The short-video gate remains:

```text
flow(A)>0, flow(D)<0, A-D>1.0, and intact person/garage through frame 38.
```

The strict action gate still fails, so do not treat any 124-frame visual-stable clip as proof of action preservation. The current RGB visual-stable grid is the meeting
visual-main for assessing structural stability; the older fixed-mix grid is archived as a historical action diagnostic.


# Current status update (2026-10-06): visual drift repair is now a fixed baseline

The main visual failure has a reproducible repair and should no longer be
mixed into action experiments. The repaired baseline is
`outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/`:

```text
RGB dual generated-history anchor
+ tail16 online visual QKV adaptation
+ original H3 chunk-endpoint latent target
+ persistent CPU raw KV
```

It keeps the person and garage coherent through 39-frame A/D clips and
124-frame W/A/D rollouts. Each 124-frame run used 8 chunks, 8 steps/chunk, 64
noisy forwards, 8 clean commits, and about 13.19 GiB CPU KV. The W run used
31,570 MiB allocated GPU peak. Playable root-level comparisons are:

```text
outputs/stage2_rgb_endpoint_visual_stable_W_124.mp4
outputs/stage2_rgb_endpoint_vs_original_W_124.mp4
outputs/stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4
outputs/stage2_rgb_endpoint_visual_stable_AD_124.mp4
```

The action gate is still intentionally failed (`flow(A)=-1.1448`,
`flow(D)=-1.4557`, `A-D=0.3109`). Therefore the next experiment must keep the
visual adapter and RGB anchor frozen, and change only the action pathway or a
real rollout-distribution matching objective. Do not run another anchor,
solver, gain, or endpoint-weight sweep, and do not call the visual-stable W
clip evidence of preserved A/D geometry. Detailed evidence is in
`outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/VISUAL_DRIFT_REPAIR_REPORT.md`.

The repaired visual adapter has also been inserted into the actual
`stage2_lite_dmd.py` critic/DMD chain. One A/D update over two chunks and four
sigmas completed without NaN/OOM (40,320 MiB GPU peak); its 39-frame A/D
rollout keeps the person intact but gives `flow(A)=-1.1419`, `flow(D)=-1.4550`,
`A-D=0.3131`. This is the Stage2 integration evidence, not an action success;
the playable artifact is
`outputs/stage2_lite_rgb_endpoint_integrated_AD_39.mp4`. Do not expand this
student to a new 124-frame action grid until a multi-state action objective
passes the short gate.


## 最新执行结果：RGB-consistent Stage2-lite v2 已完成，但 action gate 仍失败

已完成 `outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/`：39 帧、3
chunks、chunk 1/2、四个 sigma、generated history、persistent CPU raw KV，并把 student、critic、
teacher 的 previous-chunk anchor 统一为 RGB decode→H3 image encode 的 dual anchor。训练耗时
872.9 s，峰值 39.37 GiB；A/D 评估视频已转为 H.264 Constrained Baseline / YUV420P。

视觉结果明显比旧 latent-only Stage2-lite 稳定：人物和停车场结构保留到第 38 帧，人物分解问题
基本消失。但动作 gate 仍失败：

| checkpoint | flow(A) | flow(D) | A−D | `A>0,D<0,A−D>1.0` |
|---|---:|---:|---:|---|
| RGB-consistent Stage2-lite v2 | −1.1369 | −1.4556 | 0.3187 | FAIL |

本轮同时改了 anchor protocol 和 Stage2 coverage，因此它不是纯 loss ablation；它的价值是把视觉
conditioning mismatch 与 action geometry 问题分开。结论是：

```text
RGB dual anchor -> 解决后段人物分解/结构漂移
multi-chunk + multi-sigma DMD -> 一轮更新仍不足以恢复 A/D score geometry
```

因此下一步不再做 gain、anchor 或 solver sweep，也不把该 checkpoint 扩展到 124 帧。核心问题
回到已有审计结论：causal student A/D delta 的范数约为 teacher 的 6.9×，cosine 约 −0.009，
需要先对齐 action pathway/topology。

### A2 smoke 已有结果

在 RGB 协议下做了一轮 full-teacher paired delta alignment：hidden residual 的 paired cosine
均值为 0.139、norm ratio 1.974；tail4 action-QKV refiner 的 cosine 提升到 0.390、norm ratio
降到 1.018。但自由 rollout 的 flow gate 仍失败：

| Variant | flow(A) | flow(D) | A−D | Gate |
|---|---:|---:|---:|---|
| hidden residual | −1.2499 | −1.4293 | 0.1794 | FAIL |
| tail4 action-QKV | −0.8359 | −0.7685 | −0.0675 | FAIL |

这说明 score-field delta alignment 是必要但不充分的约束；当前 causal action rows/feedback
到视频 token 的 routing 仍未恢复原始 H3 的图像空间左右方向。人物和停车场在两种 variant
中都保持到第 38 帧，故当前主问题已从视觉分解转为 action geometry。

已启动 tail4 QKV 的 4-update learning curve；脚本现在在每个 round 保存
`update_XX/student_action_adapter.pt`，结束后逐轮评估 flow，避免只看 final checkpoint。

## Next Plan v4：固定 RGB 视觉协议，做最小 action-pathway alignment

### A1：冻结视觉和推理协议

统一使用：

```text
39 RGB / 12 latent / 3 chunks / chunk=5 / history=5
RGB dual anchor
8 steps/chunk / flow shift=2.22
generated history / CPU raw KV
action_prefix_mode=causal / action_feedback=true
seed=13
```

每次只改变 action-pathway 参数或 alignment objective。动作 gate 仍是：

```text
flow(A) > 0
flow(D) < 0
A-D > 1.0
人物和停车场结构在第 38 帧仍完整
```

### A2：先做 action-pathway 小容量对齐，不先扩大 Stage2

基于 `student_teacher_delta_audit.py` 的同一 generated state，训练 student 使

```text
Delta_student = v_student(A) - v_student(D)
Delta_teacher = v_teacher(A) - v_teacher(D)
```

在 RGB-consistent cache 上方向 cosine 和相对幅度都匹配。优先比较两个最小 variant：

1. action residual only：保持已有 hidden residual，加入 multi-sigma paired delta + replay；
2. action QKV refiner：只在 tail4/tail8 注入 action-conditioned Q/K/V residual，低学习率、
   零初始化，保持 RGB visual tail16 冻结。

两者都只跑 4 个 optimizer updates 的 smoke，再看 A/D 39 帧；如果 action QKV 造成人物漂移，
立即保留 action residual 结果并记录为结构冲突，不继续扩大。

### A3：用 action delta 的可学习性而非 total loss 做选择

每个 checkpoint 记录：

```text
student/teacher delta cosine
student/teacher delta norm ratio
L_replay, L_dir, L_mag
flow(A), flow(D), A-D
frame RGB MAD, boundary RGB MAD
```

只有同时满足 delta cosine 明显高于 0、norm ratio 接近 1、A/D gate 和视觉稳定，才进入下一步。
不能只看 total loss 下降，也不能用 gain 把已有错误 residual 放大后当作 geometry 恢复。

### A4：只有 action pathway 通过后才恢复 Stage2 DMD

如果 A2 在 39 帧恢复方向但 124 帧再次 drift，再把已经修正的 action pathway 放回 RGB-consistent
multi-chunk/multi-sigma Stage2，增加 update 数或 schedule/counterfactual state。若 39 帧仍无法
通过，停止 SGF/DMD 扩容，结论保持为 causal topology/action representation mismatch。

### A5：最终交付约束

在 A2/A3 通过前，不生成新的正式 124-frame W/S/A/D grid。正式主视频仍是旧的
`meeting/annotated/h3world_rgb_stable_action_grid_124_timed.mp4` 作为当前会议视觉主入口；旧 fixed-mix grid 已归档，文件和数据在
上述 2026-10-06-07 目录中。

## 当前执行结论：gain learning curve 已完成，先审计目标，再决定训练路线

RGB dual + full-attention paired teacher + FP32 action projection 的 4-step 训练和全部 A/D
39 帧评估已完成。A−D 为 **0.8272 / 0.8075 / 0.8213 / 0.8002**，未超过手工 A64/D8
初始化的 0.8286，也未通过 1.0 gate。详见
[报告](H3-World/outputs/2026-10-06-09/GAIN_CURVE_REPORT.md) 与
[可播放诊断对照](H3-World/outputs/h3world_rgb_gain_diagnostic_AD_39.mp4)。

下方“先训练 per-action gain”的计划已执行；Stage2-lite v2 的历史优先级不自动恢复。
下一轮只应先做这些可归因诊断：

1. 冻结目前 RGB-dual 视觉协议，先核查 full-teacher 对同一 generated state 的 counterfactual
   action delta。双向 prefix teacher 不等于完整 39 帧的原始 teacher，不能仅凭“关闭 causal
   mask”就认定其监督一定正确。记录 chunk/sigma 层面的 delta cosine 与 norm ratio。
2. 区分 gain 与 projection 学习：本轮两者共同更新，gain 本质上是已有 action projection
   列缩放，不增加表示能力。不要再把手工幅度放大当作新架构或最终动作学习成果。
3. 根据教师信号与梯度审计再选 action-path adaptation 或 RGB 一致的多 sigma critic/DMD；
   4 步失败不足以证明结构性不可能，也不足以证明 Stage2 必然能解决。
4. 39 帧的方向、A−D>1.0、视觉稳定必须同时通过，之后才做 action switching 和 124 帧。
   在此之前不增加新的 124-frame 主视频，不重复 gain/anchor/solver 扫描。

本轮实际产物已收敛为一份报告、逐步 checkpoint/视频和一个根目录诊断对照。最新对照是
39 帧 / 1.625 秒、2 行 A/D；不应与正式 124 帧 / 5.17 秒 grid 混淆。

full-teacher audit 已完成：在同一个 A generated state 上，3 个 chunk × 5 个 sigma 的 A/D
teacher delta 非零，但仅占平均 velocity norm 的约 1.28%–3.36%（15 个状态均值约 2.07%）。这说明
监督信号存在但很弱，不能从 4-step loss 下降推断 action geometry 已恢复。audit 原始数据见
`H3-World/outputs/2026-10-06-10/teacher_delta_audit/`。在继续训练前，应先用 D-generated
state 或多 seed 复核 delta 稳定性，并记录 student/teacher delta cosine；不要直接扩大训练
预算或再扫 inference gain。D-generated state 复核已完成：ratio 为 1.50%–4.24%，均值
2.06%，与 A-generated state 的均值 2.07% 接近。监督幅度在两个 rollout 方向都弱而非零，
但 target 的图像空间方向仍未验证；下一项应是 student/teacher delta cosine，而不是更长的
gain curve。

student/teacher delta cosine 审计也已完成（见 `outputs/2026-10-12/`）：student delta
norm 平均 69.70，teacher 10.68，norm ratio 平均 6.89×；cosine 平均 −0.0087，基本正交。
因此当前最优先级不是继续 gain，也不是马上扩 Stage2 critic，而是一个最小 action-pathway
对齐实验：固定 RGB dual、KV、solver 和 generated history，只改变 action pathway 的可见性
或一个低容量 action-token/QKV refiner，并以 student/teacher delta cosine > 0、norm ratio
接近 1 和 39-frame flow gate 作为双重验收。若 alignment 仍失败，再讨论 causal topology
重设计或 Stage2 分布匹配。

## 2026-10-06 最新状态：先恢复视觉稳定，再学习 action-dependent gain

上一版计划中的 Stage2-lite v2 暂停。最近实验发现，tail16 visual adapter 的训练协议是
`dynamic_last_frame_rgb_dual`，而后续诊断误用了 latent-only `dynamic_last_frame_dual`；这会
直接造成后段人物透明、分解。换回 RGB dual 后，39 帧人物稳定，但 fixed visual adapter
仍把 A/D 推向同一个方向。

本轮新增并验证了三件事：

1. `benchmark.py` 和 `train_online_selfrollout.py` 支持
   `--causal-adapter-scope all|commit|last_step_commit`。commit-only 不能恢复动作，说明
   visual adapter 不能简单地从 noisy score field 中移除。
2. `--paired-full-teacher` 使用原始双向 H3 在同一 generated state 上计算 A/D counterfactual
   velocity。它提供了正确的 action geometry 监督，但 1-step smoke 尚未达到 gate。
3. action residual 的单一 global gain 存在明显不对称：A 需要大增益才变为正向，D 在同样增益
   下会 ghosting。RGB dual 下的 inference-only per-action gain（A×64、D×8）得到
   `flow(A)=+0.0474`、`flow(D)=-0.7812`、A−D=`0.8286`，两条人物都保持完整；仍低于
   `A-D>1.0`，不能作为最终 checkpoint。

因此下一步只做一个有明确归因的训练改动：把 per-action gain 作为 action residual 的可训练
参数，在 RGB-dual、generated-history、8-step/chunk、full-attention H3 counterfactual teacher
下做 1→4 step learning curve。固定所有其它变量，不再扫描 anchor、solver、chunk、KV 或
重复 causal-teacher paired loss。每个 checkpoint 仍先跑 39-frame A/D；只有同时满足
`flow(A)>0`、`flow(D)<0`、A−D>`1.0` 且人物稳定，才允许重新做 124-frame grid。若 gain
训练仍停在约 0.8，接受 action representation 与 causal topology 的结构性限制，并把
RGB-dual 稳定视频作为 Stage1 feasibility 主证据；不把 inference-only gain 或普通 replay
称为 SolarWM Stage2。

## 2026-10-05 Stage2-lite 状态更新

已完成第一版真正的 critic/DMD prototype，代码为
[`H3-World/code/causal/stage2_lite_dmd.py`](H3-World/code/causal/stage2_lite_dmd.py)。它在单个共享
33B backbone 上轮换 student、critic、teacher，39 帧 / 3 chunks / 8 steps/chunk 的 4 轮实验
已完成，峰值约 39.4 GiB GPU，未发生 OOM。结果是 A=`+0.00477`、D=`-0.76839`、A-D=`0.77316`：
A/D 符号 gate 通过，但 separation 没有达到 1.0，因此不能宣称 Stage2 已解决 action geometry。

这轮实验的归因边界很清楚：critic 只训练最后 generated chunk、单个 `sigma=0.6`，每个 action
每轮只做一次 critic update 和一次 DMD student update。critic loss 约 0.09–0.11，DMD 梯度有限且
稳定，说明实现链路有效；但 signal coverage 不足以替代 SolarWM 的多 sigma、长 rollout 分布匹配。
正式结果和视频在
[`H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/RESULTS.md`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/RESULTS.md)。

## 下一步优先级（Stage2-lite v2）

只有在确实还要继续追求 action separation 时，按下面顺序扩展；不要直接扩到 124 帧：

1. 保持同一个 39-frame protocol、同一个 fixed-mix initialization、同一个 tail4 critic，先把
   critic target 从最后 chunk 扩为 chunk 1/2，并在每次 rollout 使用多个实际 solver sigma（例如
   0.94、0.79、0.57、0.24），记录每个 sigma 的 critic loss 和 DMD gradient。
2. 继续使用 generated history；critic 和 teacher 的历史 cache 都从 detached student chunks
   构建，student replay 只保留当前 chunk graph。这样仍然是真正的 Stage2-lite，不会退化成
   teacher-forcing replay。
3. 给每个 39-frame update 保存 action adapter checkpoint，并在每轮后统一跑 A/D flow。验收仍为
   `flow(A)>0`、`flow(D)<0`、`A-D>1.0`，同时检查人物、停车场结构和 boundary/frame MAD。
4. 若多 sigma/多 chunk 后 A-D 仍停在 0.8 附近，停止扩大 critic 容量，接受结论：当前问题主要是
   H3 action representation 与 causal mask 的结构性不匹配，需回到 action pathway adaptation；
   不把更多 DMD steps 当作质量修复。
5. 只有 39 帧 gate 通过后，才重新做 124-frame W/S/A/D 和 W→A→D schedule。否则正式主 demo
   继续使用现有 fixed-mix causal 结果，并把 Stage2-lite 作为“可运行但尚未恢复 geometry”的
   ablation。

执行状态（2026-10-03 23:00）：已将 Phase 1 的 paired A/D teacher-delta objective 接入
`code/causal/train_online_selfrollout.py`。训练脚本现在可在每个实际 solver sigma 上，用同一个
generated latent、同一份 student/teacher raw-KV history 同时计算 A/D replay、方向和相对幅度损失，
并按 `--checkpoint-every` 保存中间 action adapter。N1（4 steps）正在 GPU0 上运行；固定协议和其它
architecture 参数未改变。A/D 的 `prompt_embeds` 允许动作文本行不同，但 initial latent 和 audio noise
已做逐元素一致性检查。

N1 结果（4 optimizer steps，2026-10-03 23 时段）：step-04 rollout 的 A/D separation 为
`0.7687`（`flow(A)=-0.0063`、`flow(D)=-0.7750`），低于原 per-sigma replay 基线
`0.7979`，因此尚未通过方向 gate。N2 的 16-step learning curve 已启动，仍从同一
`fixed_mix_0.5/action_adapter.pt` 初始化；每 4 步保存一个 checkpoint，完成后再决定是否
停止 paired distillation 或转向 action-pathway adaptation。

N2 已完成 16 optimizer steps。四个 checkpoint 的固定 39-frame rollout 为：

| checkpoint | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| step_04 | `+0.0070` | `−0.7898` | `0.7968` |
| step_08 | `−0.0106` | `−0.7832` | `0.7726` |
| step_12 | `−0.0039` | `−0.7894` | `0.7855` |
| step_16 | `−0.0487` | `−0.7898` | `0.7411` |

训练中的 `L_dir/L_mag` 持续下降，但 rollout action geometry 没有恢复，且后期 A 的方向变差。
因此 paired score loss 在当前 action residual 路径上不能通过短片 gate；不再继续增加 paired
optimizer steps，也不进入 SGF/DMD。下一步转为低学习率 action-pathway adaptation，先做最小
可归因 smoke：冻结 causal architecture、anchor、solver 和 generated-history 协议，只解冻
H3 原 action LoRA 或 action-token refiner 的小参数集，并继续使用同一 teacher replay 与 A/D
方向验收。只有该路径在 39 帧恢复 `flow(A)>0`、`flow(D)<0`、A−D>`1.0`，才考虑更长 rollout。

## 核心目标

“证明 H3-World 能够做 causal rollout”已经完成。下一阶段不再证明 causal pipeline 是否能运行，而是固定 causal 推理协议，解决 generated history 下的 action-conditioned score field collapse：

```text
在 generated-history + 8-step causal rollout 下，恢复原始 H3 的 action geometry，先解决 A/D。
```

目标形式为：

```text
原始 H3 teacher 的 A/D action response
≈ causal student 在自身 generated history 上的 A/D action response
```

当前已知短片基线为：

| 模型/设置 | A−D horizontal-flow separation |
|---|---:|
| Original H3 30-step teacher，39 帧 | 约 `2.02` |
| causal clean-history adapter | 约 `0.96` |
| causal generated-history，无 adapter | 约 `0.015` |
| fixed-mix，8 steps/chunk，feedback on | `0.756` |
| online per-sigma replay，8 steps/chunk，4 optimizer steps | `0.798` |

这些数值是 action-response proxy，不是视频质量分数。39 帧的 A/D 分离已经部分恢复，但 A 本身仍接近零且 generated-history 画面会淡出，所以不能仅凭 A−D 上升宣称 action fidelity 已经恢复。

## Phase 0：冻结实验协议，不再改变 architecture

从 N1 开始，所有短片实验统一使用下面的协议。每次实验只允许改变 training objective，不能同时改变 anchor、chunk size、QKV block 数、solver steps 或 seed。

```text
39 RGB frames
12 latent frames
3 causal chunks
chunk size = 5 latent frames
history chunks = 5
8 steps/chunk
flow shift = 2.22
dynamic_last_frame_dual
CPU raw KV
generated history
action_prefix_mode = causal
action_feedback = true
seed = 13
same initial RGB
same prompt
same video/audio noise
```

统一初始化为：

```text
H3-World/outputs/2026-10-02-scheduled-sampling/fixed_mix_0.5/action_adapter.pt
```

固定协议的目的，是让后续曲线能够归因于 loss，而不是又一次把 solver、anchor 和 adapter 容量混在一起。

所有输出继续使用：

```text
H3-World/outputs/YYYY-MM-DD-HH/<experiment>/
```

最终对比视频才放在 `H3-World/outputs/` 根目录。

## Phase 1：Online Per-Sigma Paired Action Distillation

这是下一轮最高优先级实验。当前 online replay 已经完成：

```text
v_S(z_sigma, h_gen, A) ≈ v_T(z_sigma, h_gen, A)
```

但现在仍然是一个 action 一个 action 地匹配 teacher。普通 velocity matching 没有明确要求 student 保留 teacher 的 A/D 差异结构，因而可能让 A 和 D 共同变好，却不让两者分开。

### Paired counterfactual 条件

对于同一个 generated history `h_k`、同一个当前 noisy latent `z_sigma`、同一个 solver sigma，同时构造 A 和 D 两个 counterfactual action：

```text
a_A = A
a_D = D
```

Frozen H3 teacher：

```text
v_T^A = T(z_sigma, h_k, A, sigma)
v_T^D = T(z_sigma, h_k, D, sigma)
Delta_T = v_T^A - v_T^D
```

Student：

```text
v_S^A = S(z_sigma, h_k, A, sigma)
v_S^D = S(z_sigma, h_k, D, sigma)
Delta_S = v_S^A - v_S^D
```

同一 chunk 的 A/D pair 必须使用相同的 generated world state、相同的 noisy latent、相同的 sigma。不能用一个 A rollout 的 history 对另一个独立 D rollout 做隐式配对。

### Loss

原有 per-sigma replay：

```text
L_replay = 1/2 * ( ||v_S^A-v_T^A||² + ||v_S^D-v_T^D||² )
```

新增方向和幅度约束：

```text
L_dir = 1 - cosine(Delta_S, Delta_T)

L_mag = abs(||Delta_S|| - ||Delta_T||) / (||Delta_T|| + epsilon)
```

总损失：

```text
L = L_replay
  + lambda_dir * L_dir
  + lambda_mag * L_mag
  + lambda_boundary * L_boundary
```

首轮只固定一组 `lambda_dir`、`lambda_mag`，不做大规模超参 sweep；首轮建议以 replay 为主，使用较小的 direction/magnitude 权重，防止 action residual 为了拉大 flow 而破坏场景稳定性。具体权重、梯度范数和 checkpoint 必须写入该实验的 `training.json`。

实现要求：

1. student A/D 都在自身生成的 chunk history 上运行；
2. paired loss 使用同一 chunk、同一 sigma 的 A/D student/teacher velocity；
3. action condition 同时进入 student rollout、KV commit、student replay、teacher replay 和 pair loss；
4. 保留 `action_prefix_mode=causal` 与 `action_feedback=true`；
5. 继续使用 frozen H3 teacher，不引入 discriminator 或 GAN；
6. 第一阶段只训练已有 action residual，不重新训练 tail16 QKV；
7. 必须同时保存 `L_replay`、`L_dir`、`L_mag`、`L_boundary`、梯度范数和每步耗时。

这样做的理由是：当前异常不是单纯“视频不够真实”，而是 action-conditioned score field geometry collapse。原始 H3 teacher 已经提供了最干净的 A/D 监督，没有必要先训练一个复杂的 action discriminator。

## Phase 2：训练 16 个 optimizer steps，建立 learning curve

之前的 4 optimizer steps 仍然只是 smoke test。`0.756 → 0.798` 说明梯度方向没有完全错，但不足以判断 paired objective 是否真正有效。

第一轮正式训练使用：

```text
16 optimizer steps
A D A D A D A D A D A D A D A D
8 solver steps/chunk
```

每 4 个 optimizer steps 保存一个可回载的 action adapter：

```text
step_04/action_adapter.pt
step_08/action_adapter.pt
step_12/action_adapter.pt
step_16/action_adapter.pt
```

每个 checkpoint 都统一跑 A/D 39-frame evaluation，得到：

```text
ADSep(0), ADSep(4), ADSep(8), ADSep(12), ADSep(16)
```

并同步报告：

```text
L_replay
L_dir
L_mag
L_boundary
flow(A)
flow(D)
boundary MAD
frame MAD
人物/停车场结构的 contact sheet
```

如果训练耗时仍接近当前 8-step online 的每步数分钟，应优先保存中间 checkpoint，不能等到全部 16 steps 结束才发现 loss 或画面已经恶化。

## Phase 3：39-frame PASS gate

不要再凭主观印象决定是否扩展到 124 帧。短片必须同时满足 action fidelity 和 visual stability。

### Gate A：动作方向

必须满足：

```text
flow(A) > 0
flow(D) < 0
```

当前 `A=-0.013、D=-0.811` 虽然 A−D=`0.798`，但 A 自身还没有恢复正确符号，因此不能算通过。

### Gate B：动作分离

39-frame 原始 H3 teacher 的 A−D 约为 `2.023`。最低目标设为 teacher 的一半：

```text
A−D > 1.0
```

更理想的目标是 `A−D > 1.2`。不要求短片完全达到 `2.023` 才能继续。

### Gate C：不能用画面退化骗指标

必须人工检查和记录：

- 人物是否持续存在；
- 停车场柱子、车道线和背景结构是否稳定；
- 是否出现雾化、ghosting 或 chunk scene switch；
- boundary MAD 和 frame MAD 是否恶化；
- A/D contact sheet 是否仍然可辨认。

只有 `flow(A)>0`、`flow(D)<0`、A−D>`1.0` 和可接受的视觉稳定性同时成立，才进入 Phase 4。

## Phase 4：Counterfactual action switching

如果 Phase 1--3 通过，先不要直接扩展到 124 帧，先做 39-frame action schedule：

```text
A → D → A
D → A → D
```

在每个 chunk 边界，从同一个 generated world state `h_k` fork 两个 counterfactual action，分别运行 student 和 teacher：

```text
             history h_k
                 │
         ┌───────┴───────┐
         │               │
         A               D
         ↓               ↓
       v_S^A           v_S^D
         │               │
       v_T^A           v_T^D
```

继续匹配：

```text
Delta_S ≈ Delta_T
```

验收重点是动作切换是否发生在正确的 chunk，而不是一个 action 污染整段视频。这比整段 A 和整段 D 更接近最终 interactive world model 的使用方式。

## Phase 5：可选的 frozen action probe

只有当 paired teacher score 已经改善、但 action 仍然 collapse 时，才增加 GameGAN 风格的辅助约束。先不做 GAN discriminator。

训练一个来自原始 H3 teacher 视频的小型 frozen action probe：

```text
C(V_{k-1}, V_k) → a_k
```

覆盖至少：

```text
A, D, W, S
```

再增加：

```text
L_act = CE(C(V_hat_{k-1}, V_hat_k), a_k)
```

总损失变为：

```text
L = L_replay
  + lambda_dir * L_dir
  + lambda_mag * L_mag
  + lambda_act * L_act
```

这一阶段的目的只是验证显式 action supervision 是否能避免 collapse，不引入真正的 adversarial training，以免同时增加 teacher、critic、GAN 稳定性等变量。

## Phase 6：暂时不重新训练 tail16 QKV

当前已有证据不支持立即增加通用 QKV 训练：

```text
tail16 QKV + action residual online smoke：A−D = 0.1674
旧 tail16 QKV + fixed-mix：A−D = 0.1596
```

因此 Phase 1--5 统一只训练 action residual。等 action objective 本身通过 PASS gate 后，再比较两个小范围 ablation：

```text
A. action residual only
B. action residual + tail8 QKV
```

暂不优先尝试 tail16。tail8 更便宜，足以验证 general causal score field adaptation 是否在 action geometry 稳定之后改善视觉质量。如果 QKV 加入后 action 再次塌缩，记录为两个梯度目标冲突，而不是继续增加容量。

## Phase 7：PASS 后再制作 124-frame final grid

39-frame PASS gate 通过之前，不生成新的正式 124-frame final grid。

通过后统一生成：

```text
Original H3 30-step       New causal
W                         W
S                         S
A                         A
D                         D
```

再生成：

```text
W → A → D
A → D → A
```

配置必须固定为：

```text
124 frames
8 steps/chunk
causal action rows
action feedback ON
generated history
persistent CPU KV
same seed/noise
```

新结果与当前正式视频严格 A/B：

```text
H3-World/outputs/h3world_final_fixed_mix_action_grid_124.mp4
```

最终表格至少包含 inference time、peak GPU、CPU KV、denoiser forwards、frame MAD、boundary MAD、A/D/W/S action flow 和 schedule switching 结果。

## Phase 8：只有在这里再决定是否做 SGF/DMD

如果 39 帧方向恢复、但 124 帧仍然逐渐掉，结论就是：

```text
long-horizon generated-state distribution shift
```

这时才值得继续实现真正的 SolarWM Stage2-inspired SGF/DMD：让 student 在自己的 rollout distribution 上训练 fake score 或 distribution matching，并引入 teacher/critic 梯度。

如果 39 帧连 `flow(A)>0`、`flow(D)<0` 都恢复不了，则不要急着做 SGF。那说明问题仍在 causal action representation/pathway，应改为低学习率地联合适配：

- H3-World 原 action LoRA；
- action token refiner；
- 更早的 directed attention projections。

不能用 Stage2 蒸馏去掩盖 Stage1 action representation 本身不正确的问题。

## 未来 1--2 天只运行这五个实验

| ID | 实验 | 目的 | 成功标准 |
|---|---|---|---|
| N1 | online per-sigma + A/D paired teacher-delta loss，4 optimizer steps | 验证 paired loss 的梯度方向 | A−D 高于当前 `0.798`，并记录四项 loss |
| N2 | 同 N1，16 optimizer steps | 建立 learning curve | A−D `>1.0`，保存 step 04/08/12/16 |
| N3 | N2 最佳 checkpoint，A/D 39 帧正式验收 | 检查方向和画面 | `flow(A)>0`、`flow(D)<0`、A−D `>1.0` |
| N4 | A→D→A / D→A→D，39 帧 | 验证 action switching | chunk 切换方向正确且无 scene switch |
| N5 | N4 通过后，124 帧 W/S/A/D + schedule | 最终面试 Demo | 长时动作仍有效并报告 drift |

如果 N1/N2 完全没有提升，停止 paired distillation，转向 action-pathway adaptation。如果 N3/N4 通过但 N5 失败，再进入 SGF/DMD-like Stage2 诊断。

## 执行更新：prefix smoke 已完成，下一轮是原始 H3 LoRA tail8

N2 的 paired loss learning curve 已经证明继续增加 optimizer steps 没有恢复
generated-history 的 action geometry。随后完成的 prefix residual smoke 只让
`flow(A)` 变成 `+0.0055`，但 `flow(D)=-0.7668`、A−D=`0.7724`，仍低于 gate 的
`A−D>1.0`，所以不制作 124-frame 新 grid。

下一轮固定所有 Phase 0 推理变量，只改变 action pathway 的可训练参数：

```text
初始化：fixed_mix_0.5/action_adapter.pt
causal architecture：不变
prefix residual：可选地载入 step-04 并冻结
训练参数：H3 released LoRA，blocks 42--49 的 qkv_proj/out_proj
teacher：每次 teacher forward 恢复原始 LoRA 矩阵
学习率：先用 1e-6，4 optimizer steps smoke
```

训练脚本参数为 `--train-h3-lora --h3-lora-blocks tail8 --h3-lora-lr 1e-6`；对应的
`h3_lora_adapter.pt` 可以通过 benchmark 的 `--h3-lora-adapter` 加载。这样 teacher 不会
随着 student 的 LoRA 更新而漂移，实验仍然只改变一个训练变量。第一轮仍只跑 A/D 39 帧，
验收标准保持：`flow(A)>0`、`flow(D)<0`、A−D>`1.0`，并检查人物/停车场结构和 boundary MAD。

如果 tail8 原始 LoRA 仍不能通过 gate，下一步再考虑只训练更早的 directed attention projection；
如果短片通过，才做 A→D→A / D→A→D switching，最后才扩展 124 帧。SGF/DMD 继续等到短片
action geometry 恢复之后再决定。

### tail8 smoke 的执行结果

tail8 原始 H3 LoRA 已完成单步 smoke。`h3_lora_adapter.pt` 的 teacher swap 与 benchmark
回载均正常，但 flow(A)=`−0.0134`、flow(D)=`−0.7956`、A−D=`0.7822`，没有超过 online
baseline `0.7979`，且画面仍在约 20 帧后发生雾化。因此不继续投入 4-step tail8 训练；这条
路径目前作为负诊断保留。

此时最有信息量的下一步不是继续扩大 LoRA 容量，而是做一个更便宜的 causal action
representation 对照：使用同一 generated history 和固定 noise，对比 `action_prefix_mode`
的 `causal` 与 `all` 上界，并检查 action rows 是否在 causal mask 中丢失了跨 chunk 的控制
信息。`all` 只作为诊断上界，不作为最终方案；如果 all 上界显著恢复 A/D，说明需要修正
action-row routing；如果 all 也不恢复，则问题集中在 generated-history distribution shift，
后续才考虑 frozen action probe 或 Stage2 风格 rollout matching。

继续遵守：39-frame gate 未通过前，不制作新的 124-frame grid，不做 switching final demo，
也不把任何普通 causal solver 结果称作 SolarWM Stage2。

### routing 上界结果

`action_prefix_mode=all` 的固定协议诊断已完成。A=+0.0513、D=−0.1508、A−D=0.2021，
明显差于 causal routing 的约 0.756。未来 action rows 会污染当前视频查询，不能作为
action controllability 的上界方案；后续实验继续固定 `action_prefix_mode=causal`。

下一步的主线转为 generated-history distribution shift 诊断：保留 causal action routing，
增加一个来自原始 H3 teacher transition 的 frozen action probe，只作为辅助评价/约束候选，
先测 teacher 与 causal rollout 的 action separability 是否在视觉 drift 之前已经丢失。
如果 probe 证实 causal 视频仍含有可分辨的动作而 flow 失真，重点放在 rollout matching；
如果 probe 也无法区分 A/D，再回到 action representation 的 directed projection。仍然不在
39-frame gate 之前生成 124-frame 正式结果。

### endpoint matching 结果与 Stage2 入口

每个 chunk 最终 latent 对同 action H3 30-step latent 的 endpoint MSE（权重 `0.1`）已经做过
单步 smoke。A−D 从 baseline 的 `0.7979` 降到 `0.7459`，A 仍为负，且视频 ghosting 没有
消失；不再扫描这个权重。

因此下一阶段若继续改善视频，应实现一个明确标注为 **Stage2-inspired minimal rollout
matching** 的两阶段诊断：先 detach 学生完整 generated-history rollout，再在同一 rollout
分布上做 teacher/student score replay；报告它与当前 per-sigma online replay 的差异。这个
实验不声称复现 SolarWM SGF/DMD，也不在没有短片 gate 证据时制作 final 124-frame demo。若
该两阶段 replay 仍不能恢复 A/D，最终报告应明确指出：当前硬件和小规模 adapter 原型已
验证 causal/KV 工程可行，但完整 Stage2 需要额外 critic/score role 与更长训练预算。

## 当前明确不做

- 不下载 SolarWM 25 TB 全量数据；面试题不需要；
- 不启动官方 SolarWM 33B Stage2 全量训练；
- 不改变 Phase 0 的 architecture 和推理协议；
- 不把普通 8-step solver 称为 SolarWM Stage2；
- 不继续增加 RGB anchor、soft overlap 或静态 history mix 作为默认方案；
- 不在 39-frame PASS gate 之前训练 124-frame online student；
- 不把 A−D 单一指标当作视频质量，需要同时报告 action fidelity 和 visual stability。

## 相关文件与现状

- 项目总体记录：`progress.md`；
- 面试说明和最新实验摘要：`H3-World/README.md`；
- causal benchmark：`H3-World/code/causal/benchmark.py`；
- online replay 实现：`H3-World/code/causal/train_online_selfrollout.py`；
- action flow proxy：`H3-World/code/causal/evaluate_action_control.py`；
- 当前 124-frame grid：`H3-World/outputs/h3world_final_fixed_mix_action_grid_124.mp4`；
- 当前 8-step online 基线：`H3-World/outputs/2026-10-03-21/online_sigma_feedback_39_8x4_eval/`；
- SolarWM Stage2 源码参考：
  `SolarWM/src/solarwm/backends/minimax_h3/sgf_rollout.py`、
  `SolarWM/src/solarwm/backends/minimax_h3/stage2_runtime.py`。

### two-pass replay 结果

最小 two-pass replay（weight=`0.1`, sigma=`0.6`）已经完成单步 smoke，A=−0.0008、
D=−0.7551、A−D=0.7543，低于 per-sigma baseline `0.7979`，没有消除后段 ghosting。它
验证了“把最终 generated chunk 再 replay 一次”本身不足以恢复 action geometry；SolarWM 的
Stage2 关键在额外 critic/score role 和 rollout-distribution gradient，而不是多调用一次
frozen teacher。

因此项目进入收敛阶段：不再堆叠新的 anchor、mask、LoRA 或 endpoint loss。保留当前最佳
39-frame baseline、所有失败 ablation 和可复现实验日志；完善 final report，明确区分
Stage1 causal/KV feasibility、action-preservation 的未通过 gate，以及需要完整 Stage2
训练才能解决的 generated-history drift。124-frame 正式视频仍只作为现状对照，不宣称新
causal checkpoint 已达到题目验收标准。

## 2026-10-06 最新执行结论：tail4 QKV 4-update curve 也未通过 action gate

在固定 RGB dual 视觉协议后，完成了 tail4 action-QKV paired full-teacher alignment 的 4 个
update，并逐轮生成 A/D 39-frame 自回归视频。逐轮结果为：

| update | flow(A) | flow(D) | A-D | paired cosine | norm ratio | gate |
|---:|---:|---:|---:|---:|---:|:---|
| 1 | -0.8563 | -0.7790 | -0.0774 | 0.390 | 1.018 | FAIL |
| 2 | -0.8886 | -0.7536 | -0.1350 | 0.347 | 0.973 | FAIL |
| 3 | -0.8703 | -0.7785 | -0.0918 | 0.345 | 1.019 | FAIL |
| 4 | -0.8506 | -0.7684 | -0.0822 | 0.424 | 1.045 | FAIL |

四个 checkpoint 都保持人物与停车场结构到第 38 帧，说明 RGB anchor 的视觉协议是稳定的；但
水平 flow 的 A/D 符号和 separation 都没有恢复。paired score-field cosine 与 norm ratio
有所改善，却没有转化为 free generated-history rollout 的 image-space action response。

因此停止这条 tail4 QKV 4-update 曲线，不再用更多 update 试图解决同一个问题。下一步选择应
从“继续训练 loss”转为“定位 causal action routing/topology”：

1. 保留 RGB anchor、KV cache、solver、chunk 和 seed，不再做 inference gain 或 anchor sweep。
2. 用最小的 action-token/action-row pathway probe 对照原始双向 H3，确认 action rows 在 causal
   packed sequence 中是否仍连接到当前 chunk 的 video token；记录每层 action-to-video attention
   或 hidden sensitivity，而不是只看最终 flow。
3. 若路径本身被 causal mask 截断，优先实现局部 action-to-current-chunk bypass/refiner，再用
   同一 39-frame gate 验证；这比继续扩大 Stage2 critic 更有归因价值。
4. 只有路径 probe 和短片 action gate 同时通过，才回到 Stage2 distribution matching 并扩展
   124 帧。当前不要生成新的正式 124-frame grid。

完整 curve 报告位于：

```text
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_ALIGNMENT_LEARNING_CURVE.md
```


## 2026-10-06 action-routing probe：feedback edge 存在，但贡献很小

新增 `H3-World/code/causal/probe_action_routing.py`，在同一 generated A history、同一 chunk 1
noisy state、同一 RGB dual anchor 和 sigma=0.6 上，比较四种 prefix/feedback 策略。A/D score
delta norm 为：

| variant | A/D delta norm | relative |
|---|---:|---:|
| own + feedback off | 5.502 | 0.748× |
| causal + feedback off | 7.352 | 1.000× |
| causal + feedback on | 7.642 | 1.039× |
| all + feedback on | 8.338 | 1.134× |

`causal_fb1` 与 `causal_fb0` 明确不同，证明 action-row -> current-video 的反馈边确实在工作；
因此不再优先实现未经证实的 bypass。相对于约 643 的总 velocity norm，动作 delta 只有约 1.2%，
与 teacher delta audit 的弱信号一致。下一步优先级调整为：

1. 保持当前 causal mask/KV/RGB anchor 不变，检查原始 H3 action LoRA 的 action embedding、action
   row hidden 和 video output 对 A/D 的 sensitivity；确认是否在 causal rollout 中被 score field
   旋转或压弱，而不是继续改 mask。
2. 如果 representation sensitivity 在单 chunk 仍存在但多 chunk 后消失，再做 generated-history
   distribution matching；如果单 chunk 就没有正确方向，回到 action representation adaptation。
3. 只有 action probe 与 39-frame flow gate 同时改善，才进入 124-frame；当前不生成新的正式 grid。

详细 probe 记录：`H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_ROUTING_PROBE.md`。


## 2026-10-06 action-geometry probe：幅度接近但方向正交

在同一个 generated A history chunk 0 -> chunk 1、同一 noisy latent、RGB dual anchor 和 sigma=0.6
上，比较冻结 visual causal adapter 与原始双向 H3 teacher 的 A/D counterfactual delta。结果：

| quantity | causal | original H3 teacher |
|---|---:|---:|
| A/D delta norm | 7.642 | 8.704 |
| causal/teacher norm ratio | 0.878 | — |
| delta cosine | **−0.015** | — |

动作 delta 的幅度已经接近 teacher，但方向几乎正交。因此问题不是 action signal 完全消失，也不是
简单增大 gain 就能解决，而是 causal chunk + generated history 改变了 action-conditioned score
field 的方向。下一步不再做 action-feedback bypass 或 gain sweep；如果继续，只做一个有明确
归因的 action representation/score-field adaptation，随后立即用同一 39-frame gate 验证。

详细记录：`H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_GEOMETRY_PROBE.md`。


## 2026-10-06 action-prefix 1-step smoke 未通过 gate

冻结 video action residual、visual RGB adapter，只训练 tail8 action-prefix hidden residual 的 1-step
结果：`flow(A)=-1.1091`、`flow(D)=-1.4603`、`A-D=+0.3513`，人物保持但严格 gate 失败。它比
tail4 QKV 的近零 separation 略有区分度，但 A 仍为负，不能说明 action geometry 已恢复。

因此不直接扩展 prefix 到 4/16 steps。若还要继续，必须先改变监督：使用多 generated states/seeds
或显式 image-space action probe；单一短 rollout 的 full-teacher score delta 不足以约束正确的
图像运动方向。否则项目应收敛为：causal/KV 工程可行，RGB anchor 可稳定视觉，action geometry
仍未恢复，完整 Stage2 需要更强的 rollout-distribution/action supervision。


## 收敛决策：三类 action-path adaptation 都未通过短片 gate

已完成并评估：

| route | A-D | 结果 |
|---|---:|---|
| tail4 action-QKV paired, 4 updates | -0.077 到 -0.135 | A/D 同向，FAIL |
| tail8 action-prefix, 1 update | +0.351 | A、D 都负，FAIL |
| released H3 action LoRA tail8, 1 update | +0.290 | A、D 都负，FAIL |

结合冻结 geometry probe（causal/teacher delta cosine=-0.015）和 routing probe（feedback edge 存在但
只改变约 3.9% delta norm），继续增加同类 optimizer steps 没有足够归因价值。项目现在收敛为：

- causal chunk attention + persistent raw KV + RGB-consistent anchor 在工程上可运行；
- 视觉 anchor mismatch 修复后，39/124 帧 rollout 的人物/场景稳定性可以报告；
- generated-history causal score field 的 action geometry 尚未恢复，不能声称保留 H3 action control；
- 真正解决需要多状态、多 seed 的 action supervision 或 SolarWM Stage2 级 rollout-distribution
  matching，而不是继续在单场景单 state 上堆 LoRA。

不再生成新的正式 124-frame action grid。最终面试交付应明确区分 Stage1 causal/KV feasibility、
action-preservation gate 未通过、以及需要更大训练/数据才能解决的结构性限制。

## 最终收敛说明

`H3-World/outputs/2026-10-06-08/FINAL_ACTION_DIAGNOSTIC.md` 汇总了全部新增路线。当前面试题可
严谨交付的结论是：causal chunk attention、persistent raw KV、RGB-consistent anchor 和长视频
执行链路已经验证；但 generated-history causal score field 的 action geometry 尚未恢复，不能
声称原始 H3 action control 已保真保留。若后续重新启动，需要多状态/多 seed action supervision
或完整 Stage2 distribution matching，不能再用单场景 single-state adapter update 作为替代。
