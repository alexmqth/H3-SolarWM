# 5 分钟现场讲稿

## 0:00–0:35：先解释为什么早期画面会崩

“先说明一个展示问题：会议包第一版播放的是旧的 fixed-mix action grid。它没有 visual tail16 adapter，而且推理使用 latent-only dual anchor；训练和推理的 anchor protocol 不一致，所以后半段会出现人物透明、分裂和车库 tearing。那个 MP4 能解码，只说明容器和编码没坏，不说明视觉质量合格。这里是我选片和标注的问题。现在主视频换成了后续 RGB-consistent anchor 加 visual QKV adapter 的结果。”

## 0:35–1:05：结论与主视频

打开 `annotated/h3world_rgb_stable_action_grid_124_timed.mp4`。

“我的问题是：SolarWM 的 causal chunk、KV cache 和少步训练思路能不能迁移到 H3-World，同时保留 action control？结论分两层。工程链路已经跑通：H3-World 可以按 chunk causal rollout，persistent raw KV 可以复用历史，124 帧视频能完整生成。RGB-consistent anchor 明显减少了人物和场景分解。但 generated-history 下 A/D 的 image-space action geometry 还没有恢复，所以我不会把它说成完整 action-preserving Stage2。”

## 1:05–1:50：H3 和 SolarWM 的连接

“H3-World 的关键是 action rows 和 directed action routing。改造保留这些条件，只把视频 token 按 5 个 latent frames 分 chunk；每个 chunk 完成后用 clean latent 做一次 commit，把每层 raw K/V 写入 CPU history。下一个 chunk 读取历史 K/V，只计算新 token。右侧视频是 8 steps/chunk，8 个 chunks，因此是 64 次 noisy forwards 加 8 次 clean commits，并不是整段视频只调用 8 次网络。”

“这对应 SolarWM Stage0.5/Stage1 的 causal interface 和 teacher replay。Stage2 是另外的问题：student 在自己的 generated history 上训练 fake score，再和 frozen teacher 做 distribution matching。KV cache 解决历史复用，不会自动把 30 steps 变成 8 steps。”

## 1:50–2:35：说效率和连续性

打开 `METRICS.md`。

“原始 H3 30-step 的单次端到端记录是 441.5–454.2 秒；当前 RGB-main causal 是 673.4–767.9 秒。首块 41.8–57.4 秒，平均 chunk 75.1–88.6 秒；GPU 峰值约 30.8–39.0 GiB，CPU raw KV 约 13.19 GiB。这里是单次 recorded run，没有 warmup 后多次均值，权重、KV 和 activation 也没有单独拆分。”

“因此当前 prototype 不是总 wall-clock 加速结果。它的价值是 causal execution、历史复用和可增量提交；要真正减少总耗时，需要训练少步 student，并降低每个 chunk 的 solver evaluations。”

## 2:35–3:20：播放视觉修复和 Stage2-lite

打开 `visual_stability/stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4`。

“旧协议把生成的 latent tail 直接 patchify 成第二 anchor，和训练时的 RGB decode/re-encode 语义不一致。统一为 RGB decode、再经过 H3 image branch encode 的 dual anchor 后，39 帧和 124 帧的末尾人物、车库结构明显更完整。这里的改善同时伴随 visual adapter、anchor 和 routing 配置变化，所以我把它报告为修复后的 protocol，而不是声称一个单变量因果证明。”

打开 `stage2_lite/stage2_lite_rgb_endpoint_integrated_AD_39.mp4`。

“Stage2-lite 的 student self-rollout、fake-score critic 和 frozen teacher 链路也能运行，没有 NaN/OOM；但短片 action gate 仍然失败，因此这只是 SolarWM Stage2 核心角色的 feasibility evidence，不是官方 Stage2 复现。”

## 3:20–4:20：动作结果

“当前主视频的 Farneback 水平光流是：原始 H3 的 A=`+1.077`、D=`-1.602`，A-D=`2.679`；RGB-main causal 的 A=`-0.784`、D=`-1.007`，A-D=`0.223`。所以 causal 视频不是简单静止，也不是完全相同的四条片，但 A 的方向符号错了，严格 gate `A>0、D<0、A-D>1.0` 没通过。视觉稳定性恢复和 action preservation 是两个独立问题。”

“早期 fixed-mix 表里 A-D=`0.453`，那是旧协议的历史诊断，不是当前主视频的结果；我已经把它移到 diagnostics，避免把两个实验混在一张表里。”

## 4:20–5:00：收束

“最终结论是：H3-World 的 causalization、persistent KV 和 clean commit 在真实 33B 模型上可行；RGB-consistent anchoring 能显著改善长时视觉稳定性。但 generated-history 改变了 H3 原始 action-conditioned score geometry，当前还没有证明四方向完全保真。下一步应做多 state、多 seed 的 counterfactual action supervision，或者完整的 SolarWM Stage2 rollout-distribution matching，而不是继续堆单一 anchor、gain 或 solver sweep。”
