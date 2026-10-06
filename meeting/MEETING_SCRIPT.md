# 5 分钟现场讲稿

## 0:00–0:30：先说结论

“我的问题是：SolarWM 的 causal chunk、KV cache 和少步训练思路能不能迁移到 H3-World，同时保留它原来的 action control？结论分两层。工程上已经跑通了：H3-World 可以按 chunk causal rollout，历史可以用 persistent raw KV 复用，124 帧视频能完整生成。视觉稳定性也已经通过 RGB-consistent anchor 修复。可是 generated-history 会改变 H3 的 action-conditioned score geometry，A/D 方向还没有达到原始 H3 的保真 gate，所以我不会把这个 prototype 说成完整 action-preserving Stage2。”

## 0:30–1:10：播放主视频

打开 `annotated/h3world_final_action_grid_124_timed.mp4`。

“这里左边是原始 H3-World 的 30-step full-horizon inference，右边是 causal prototype。四个动作都固定了同一首帧、prompt、seed、初始 noise 和 124 帧长度。右侧的 8 steps/chunk 是每个 chunk 的步数，不是全片只调用 8 次；这里总共是 8 chunks、64 次 noisy forwards，另外有 8 次 clean KV commit。标题下面的时间是日志中的 recorded end-to-end wall time。”

## 1:10–1:50：解释 H3 和 SolarWM 的连接

“H3-World 本身的关键是 action rows 和 directed action routing。我的改造保留了这些 action rows，只把视频 token 按 5 个 latent frames 分 chunk；前一个 chunk denoise 完后，用 clean latent 做一次 commit，把每层 raw K/V 写入 CPU history cache。下一个 chunk 只计算新 token，并读取历史 K/V。这样 causal mask、KV reuse、action prefix 和 H3 image condition 在一条真实 33B 模型上同时工作。”

“这对应 SolarWM Stage0.5/Stage1 的 causal interface 和 teacher replay。Stage2 是另外一个问题：student 要在自己的 generated history 上训练 fake score，再和 frozen teacher 做 distribution matching；KV cache 本身不会自动减少采样步数。”

## 1:50–2:40：说效率和连续性

打开 `METRICS.md`。

“正式 fixed-mix grid 中，原始 30-step 端到端约 441–454 秒，causal 8-step/chunk 约 383–452 秒。causal 首块大约 37–42 秒，之后每块大约 39–48 秒；GPU peak 约 39.9 GiB，5-chunk history 的 CPU raw KV 约 13.5 GiB。这里要诚实说明，这些是每个 action 的单次 recorded run，不是 warmup 后多次均值。”

“当前 causal prototype 不是总 wall-clock 的加速结果，因为 30 次 full-horizon forwards 对比的是 64 次 chunk forwards。它的价值先是因果执行、历史复用和可交互接口。要进一步降总耗时，需要让每个 chunk 真正只用少量 student steps，这就是 Stage2 的作用。”

“连续性方面，表中的 mean RGB MAD 和 boundary MAD 显示 causal 的 A/D 边界尖峰更大，说明生成历史漂移还存在。”

## 2:40–3:30：播放视觉修复和 Stage2-lite

打开 `visual_stability/stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4`。

“早期人物分解不是编码问题，而是 anchor protocol mismatch。训练 visual adapter 时是 generated prefix 解码成 RGB，再过 H3 image branch；旧推理却直接把 latent tail patchify 成第二 anchor。统一成 RGB decode/re-encode 的 dual anchor 后，39 帧和 124 帧的人物、车库结构都可以保持到末尾。”

打开 `stage2_lite/stage2_lite_rgb_endpoint_integrated_AD_39.mp4`。

“我又把这个 visual adapter 接入了 Stage2-lite 的 student self-rollout、fake-score critic 和 frozen teacher 链路。训练 finite、没有 NaN/OOM，说明 Stage2 核心角色在当前硬件上可以做最小原型。但是一轮 DMD surrogate 没有恢复动作，所以我把它当 feasibility evidence，不把它写成官方 Stage2 复现。”

## 3:30–4:20：解释 action control 结果

“原始 teacher 的 A-D 水平光流分离约 2.679；formal causal fixed-mix grid 约 0.453。A 的 flow 从 +1.077 降到 +0.143，D 从 -1.602 降到 -0.311。也就是说 causal 输出不是完全与 action 无关，但 action geometry 明显变弱，边界连续性也变差。”

“在更严格的 RGB generated-history 39-frame gate 中，我要求 A>0、D<0、A-D>1.0，同时人物和车库完整。这个 gate 目前没有通过。routing all、STILL counterfactual、teacher/causal delta probe 和多种小 action adapter 都没有解决；frozen causal 与 teacher 的 delta cosine 约为 -0.015，说明不是 action signal 简单消失，而是方向改变了。”

## 4:20–5:00：收束和后续

“所以我的最终结论是：H3-World 的 causalization 工程上可行，persistent KV 和 clean commit 真实工作，RGB anchor 解决了长时视觉分解；但 generated-history 下的 action-conditioned score geometry 还没有恢复。下一步不会继续做单 state 的 gain 或 anchor sweep，而是做 multi-state/multi-seed counterfactual action supervision，或者完成更完整的 SolarWM Stage2 rollout-distribution matching。这个实验已经把问题从‘能不能 causalize’收敛到了‘怎样恢复 action geometry’。”
