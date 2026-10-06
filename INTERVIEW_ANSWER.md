# 面试题回答：在 H3-World 中验证 SolarWM 的因果少步生成

## 0. 会议主视频的纠错说明（2026-10-07）

第一次整理会议包时把旧 fixed-mix action grid 当作主视频。那套运行没有 visual tail16 adapter，并使用 latent-only dual anchor；它可以解码，但 generated-history 后段会出现人物分解和车库 tearing。当前主视频改为 `meeting/annotated/h3world_rgb_stable_action_grid_124_timed.mp4`，使用 RGB-consistent dual anchor + tail16 visual QKV adapter。视觉结构明显更稳定，但 A/D flow 仍未通过 gate；这次更换同时改变了多个协议，因此只能报告为组合协议的视觉修复，不能把改善归因于单个 anchor 或 adapter。

当前 RGB-main 的 124-frame A/D flow 为 A=`-0.784`、D=`-1.007`、A-D=`0.223`，原始 H3 为 A=`+1.077`、D=`-1.602`、A-D=`2.679`。因此 124 帧的视觉改善不等于 action control 恢复；提交包明确保留这个失败结果。

最新长度测试见 [10/20 秒报告](meeting/long_horizon/README.md)：同一 RGB checkpoint 在 10 秒后段已出现 blur/ghosting，20 秒约 10 秒开始严重雾化，15 秒后人物和场景难以辨认。**20 秒视觉稳定性失败**；下文 39/124 帧的改善不能外推为长时稳定。旧版动作响应较强但画面漂移的完整对比也保存在 [三列 A/D demo](meeting/action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4)。

## 1. 项目目的

H3-World 把初始图像、场景 prompt 和每个 latent 时间位置的键盘 action 转成 action-conditioned 视频。原始 H3-World 推理使用双向/完整视频上下文的 30-step 生成；它有 action control，但不能直接把已经生成的 chunk 当作历史流式复用。

SolarWM 的核心启发是把长视频拆成 causal chunks，历史 chunk 通过 KV cache 复用，并用 Stage2 的 generated-history distribution matching 解决少步 student 在自己历史上的分布漂移。本项目不是复现 33B SolarWM Stage2，而是回答四个问题：

1. H3-World 能否被改成可运行的 causal chunk rollout？
2. persistent KV cache 是否真的工作？
3. RGB anchor 能否解决 generated-history 的视觉分解？
4. 原始 H3 的 A/D action geometry 是否仍然存在？

## 2. SolarWM 三个阶段

### Stage0.5

Stage0.5 是从全序列/非因果模型到因果分块模型的结构准备阶段。模型仍然可以使用较完整的训练监督，但 attention、chunk 边界、历史 KV 和 action/condition 的时间路由需要变成 causal-safe 形式。它让模型具备按 chunk 生成和提交历史状态的接口，并不等于少步蒸馏。

本项目对应的是 H3-World 的 block/chunk causal attention、action row causal routing、raw K/V 分块提交和 clean KV commit。

### Stage1

Stage1 是在 causal 架构上做 teacher-forced 或 per-sigma teacher replay 的适配。原始 H3 teacher 对当前 chunk 的 noisy latent/velocity 提供监督，student 学到 causal score field。它能验证 causal pipeline 和训练接口，但如果训练历史主要来自 teacher 或 clean history，student rollout 看到的 generated history 仍然不同。

本项目的 Stage1-style 原型包括 5 latent frames/chunk、5-chunk sliding history、每 chunk 8 solver steps、persistent raw KV、generated-history self-rollout、RGB dual anchor、per-sigma teacher replay，以及 paired A/D delta 诊断。

### Stage2

Stage2 让 causal student 在自己的 generated trajectory 上训练。SolarWM 的 SGF/DMD 类方法通常保留 frozen bidirectional teacher，同时训练 fake-score/fake-distribution model 去描述 student 当前生成分布，再用 teacher score 与 fake score 的差异形成 distribution-matching gradient。它解决的是 exposure bias、generated-history drift 和少步 student 分布偏移，而不是 KV cache 本身。

本项目实现的是 Stage2-lite feasibility chain：

    student self-rollout
        -> detached generated history
        -> trainable fake-score critic
        -> frozen teacher score
        -> DMD surrogate update

它共享一个 H3 33B backbone，轮换 student/critic/teacher adapter，不是官方 SolarWM Stage2 recipe，也没有声称完成 SGF/DMD 的完整复现。

## 3. Stage2 为什么能减少采样步数

KV cache 只减少历史 token 的重复计算，不会改变每个 chunk 需要多少次 denoiser evaluation。Stage2 的少步收益来自训练 student，使它在少量 solver steps 中近似原始多步 teacher 在目标生成分布上的更新方向。teacher/fake-score distribution matching 让 student 学会自己的 generated-history 状态下的有效 score，而不是只在 teacher-forced clean history 上拟合。

因此要区分：

    causal mask + KV cache = 可流式生成、复用历史计算
    Stage2 SGF/DMD = 训练少步 student 的分布匹配

本项目的 8 steps/chunk 是每个 chunk 的少步诊断，不是 8 次 forward 生成完整 124 帧。124 帧配置是 8 chunks x 8 noisy forwards = 64 次 noisy denoiser forward，另有 8 次 clean KV commit。

## 4. H3-World 原始流程

原始 H3-World 的推理大致是：

1. 输入一张 initial RGB image、scene prompt、随机 noise/audio condition 和一组 action preset；
2. 把 W/S/A/D 等 action 转成每个 latent 时间位置的 action language rows；
3. 通过 H3 directed attention routing，把 action row 绑定到对应 future video latent；
4. 对完整目标视频做 30-step flow/diffusion-style denoising；
5. VAE 解码生成视频。

原始流程的优势是 action-conditioned score field 由完整模型直接建模；缺点是长视频流式生成时历史会重复计算，而且没有严格的 chunk-wise causal commit。

## 5. H3-World 与 SolarWM causal rollout 的差异

| 维度 | 原始 H3-World | 本项目 causal prototype |
|---|---|---|
| 时间上下文 | 完整视频 score evaluation | 当前 chunk + 历史 clean KV |
| 视频组织 | 一次性完整 horizon | 5 latent frames/chunk，逐块提交 |
| 历史计算 | 重复计算 | persistent raw KV，CPU offload |
| action rows | H3 directed action routing | causal action prefix + 可选 action feedback |
| 训练历史 | teacher/完整序列 | generated-history self-rollout 诊断 |
| anchor | 固定 initial image branch | 固定 initial anchor + generated tail 的 RGB dual anchor |
| 少步训练 | 原始 30 steps | 8 steps/chunk 的 Stage1/Stage2-lite 诊断 |
| 目标 | H3 action control | 验证 causal 可行性并审计 action geometry |

## 6. persistent raw KV 和 clean commit

H3ChunkCache 保存每层的 raw K/V。每个 chunk denoise 完成后，以 clean latent 做一次无噪声 forward，把得到的 K/V 提交到 history cache；下一个 chunk 只对新 token 做计算，同时读取历史 K/V。历史缓存放在 CPU，避免 33B 模型和长序列把 GPU 显存耗尽。

clean commit 很重要：如果把某一个中间 noisy solver state 写入历史，后续 chunk 会把噪声状态当成真实世界历史，误差会累积并污染 action/visual conditioning。clean commit 也使 replay audit 可以比较写入 cache 前后的结果；本项目 replay error 为 0。

## 7. RGB anchor 为什么修复人物分解

早期 Stage2-lite 的 visual adapter 按下面协议训练：

    generated prefix -> VAE decode RGB -> last RGB frame
    -> H3 image branch encode(process_image=True)
    -> second dual anchor

但旧推理把 generated latent tail 直接 patchify 后作为第二 anchor。这是训练/推理 protocol mismatch，导致后段人物透明、分裂和 ghosting。修复后训练和推理都使用 RGB decode/re-encode 的 dynamic_last_frame_rgb_dual，再结合 tail16 visual QKV adaptation 和 endpoint target。39 帧 A/D 以及 124 帧 W/A/D 都能保持人物和停车场结构到视频末尾。

这只证明视觉稳定性改善，不证明 action direction 恢复。

## 8. action control 的验收和结果

固定同一 initial image、prompt、seed/noise 和 causal checkpoint，使用 39 RGB / 12 latent / 3 chunks、5 latent frames/chunk、8 steps/chunk、RGB dual、generated history、CPU raw KV、action_prefix_mode=causal、action_feedback=true。

严格 gate 是：

    flow(A) > 0
    flow(D) < 0
    A-D > 1.0
    person and garage remain intact through frame 38

原始 H3 teacher 约为：

    flow(A) = +1.181, flow(D) = -0.842, A-D = 2.023

RGB visual baseline 约为：

    flow(A) = -1.1535, flow(D) = -1.4557, A-D = 0.30

最新 corrected own-history endpoint + paired QKV：

    flow(A) = -0.8086
    flow(D) = -0.6893
    A-D     = -0.1194
    latent delta cosine = 0.0655

因此不能写“动作仍然有效”或“完整保真”。可以写：causal rollout 的工程链路已经跑通，但 generated-history causal score field 的 A/D 几何尚未恢复。

## 9. action failure 的定位

已经排除了几种简单原因：

- action_prefix_mode=all 让未来 action rows 可见，仍为 flow(A)=-1.1535、flow(D)=-1.4733、A-D=0.3198，所以只是扩大 action 可见性不能修复方向；
- STILL counterfactual 为 -0.7129，从 A/D 中减去 STILL 后仍不能得到正确左右方向，不能归因于共同场景漂移；
- frozen causal 与 original teacher 的 A/D delta cosine 约为 -0.015，norm ratio 约 0.878，说明 action signal 幅度并未简单消失，而是方向几乎正交；
- hidden action residual、action-prefix residual、released H3 action-LoRA、tail4/tail8 action-QKV 和 Stage2-lite DMD 都没有通过短片 gate。

最可信的解释是：causal attention + generated-history 改变了原始双向 H3 的 action-conditioned score geometry；单一 state 的 gain、anchor 或小 LoRA 不能把它旋回 image-space 左右方向。

## 10. 面试题最终结论

本项目回答了两层问题：

1. **能否 causalize？可以。** H3-World 已经能按 chunk 使用 causal attention、persistent raw KV、clean commit 和 generated-history rollout，39/124/243/481 帧均完成执行和视频解码。
2. **能否证明原 action control 完整保留，以及长视频稳定？不能。** RGB 协议改善了 124 帧视觉结构，但 A/D action geometry 未通过严格 gate，481 帧出现严重视觉崩坏。执行完成与视觉质量是不同验收项。

因此更准确的研究结论是：

    Causalization with persistent KV is mechanically feasible. Visual adaptation
    improves coherence at 124 frames, but the 20-second rollout collapses and
    the strict A/D action gate remains unmet. Multi-state/multi-seed action
    supervision and stronger rollout-distribution training are possible next
    steps; neither long-video stability nor full action preservation is established.
