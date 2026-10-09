# H3-World × SolarWM：一分钟汇报导航

**目标：** 将SolarWM的因果分块、KV cache与少步思路迁入H3-World，改善长视频效率，同时保留action control。

**当前结论：** strict causal + persistent KV已在真实33B上跑通；RGB联合协议在124帧改善视觉，却没有恢复A/D控制，20秒仍失败。最新V3从Original恢复原生条件、使用Same-σ和C12→5联合重算，在**第二块**同时取得局部动作与人物结构正结果。还没有“动作对、长期稳、整体更快”的完整模型。

**建议播放顺序：**

1. [Original vs V1：直接因果化丢失了什么](00_comparison_gallery/V1_vs_original.mp4)。
2. [V1→V2：视觉联合修复与动作局限](00_comparison_gallery/V2_vs_V1.mp4)。
3. [V3四路径：自身history后的局部正结果](00_comparison_gallery/V3_four_paths_56.mp4)，黄色边框从RGB39开始。
4. [V2→V3：前56帧跨协议对比](00_comparison_gallery/V3_vs_V2.mp4)。V3没有继承V2的visual adapter。
5. [保留的20秒失败证据](V2_rgb_anchor/videos/V2_long20s_failure.mp4)。后段完整，没有裁掉崩坏。

| 版本 | Action fidelity | Visual stability | 长时 rollout | Persistent KV | Sampling steps | 新增训练 | 验收状态 |
|---|---|---|---|---|---|---|---|
| V0 Original | A/D 正控；W/S 以视频为准 | 124f 基本完整 | 有243/481f参考；仍有几何变形 | 否 | 30整段；另存历史50步 | 无，released LoRA | Reference，不是GT |
| V1 Native causal | A/D显著减弱；本代表A符号错 | 停滞、透明/重影等退化 | 124f执行完成 | 是，CPU raw video KV | 8/chunk × 8 + 8 commits | 本代表无 | 工程可行，质量/动作未过 |
| V2 RGB联合修复 | A符号错；未恢复 | 124f相对改善；20s失败 | 124/243/481f均有片 | 是，CPU raw video KV | 8/chunk | visual QKV + action residual | 视觉局部改善，联合验收失败 |
| V3 Same-σ C12→5 | 两份自身history下当前A/D方向正确 | 第二块人物结构保持 | 仅56f、2块 | **否**，T2逐sigma联合重算 | 30/chunk | 无，从Original重新出发 | **Current Best Local Causal Rollout Candidate** |
| V4 Efficient causal | 待验证 | 待验证 | 待验证 | 目标：是 | 先30/chunk可信，再少步 | 待定 | Planned，无checkpoint/视频 |

## 按需展开

- [V0 Original](V0_original/README.md) / [V1 Native causal](V1_native_causal/README.md) / [V2 RGB联合修复](V2_rgb_anchor/README.md) / [V3 Same-σ](V3_same_sigma/README.md)：统一十项说明、核心源码与原片。
- [完整横向比较画廊](00_comparison_gallery/README.md)：包括逐A、逐D和合并片；无新模型采样。
- [研究路线](roadmap.md) / [三个研究分支与失败实验](01_research_branches_summary/README.md)。
- [下一步与Go/No-Go](02_next_steps/README.md) / [5分钟答辩稿](TALK_5MIN.md)。
- [单次历史性能与口径](METRICS.md) / [对比公平性与缺失](COMPARISON_PROTOCOL.md)。

**当前最好的是V3局部候选，不是最终高效模型。** 下一步缺的是将这项局部能力迁回strict causal/KV，再验证多块，随后才有可靠的AnyFlow与on-policy DMD基础。本次只整理现有结果，没有启动这些研究。

本目录可独立复制用于汇报：视频均为实际文件，源码是阅读快照，不含33B权重。大规模原始证据留在完整仓库；各版本PROVENANCE.json记录源位置/配置/hash。所有时间为原共享主机单次记录，没有宣称warmup多次均值或公平speedup。
