# H3-SolarWM：研究报告与视频导航

**正式参考：V3 Baseline，30步、124帧的严格因果与持久KV可行性版本。** 画质与连续性仍为PARTIAL。2026-10-11起按以下研究问题整理；每个尝试先说明名称含义和目的，再列结果与视频。

| 版本 / 路线 | 目的 | 报告入口 |
| --- | --- | --- |
| V0 / V1 | 原始双向参考 / 初次因果化 | [V0](V0_original/README.md)、[V1](V1_native_causal/README.md) |
| V2 | 修复视觉和动作退化的两条路线 | [RGB Anchor / Same-σ](v2/README.md) |
| V3 Baseline | 因果缓存下同时保留动作和基本结构 | [正式124帧基线](v3/v3_baseline/README.md) |
| V3 Sliding Window | 固定历史KV容量，比较全局/局部位置 | [Global / Local](v3/sliding_windows/README.md) |
| V3 FM → AnyFlow | 普通减步、AnyFlow训练、场景迁移和数据准备 | [FM-30NFE/8/4、AF-8NFE/4](v3/fm2anyflow/README.md) |
| V4 SGF / DMD | 在学生自身生成样本上进行分布匹配蒸馏 | [DMD尝试及负结果](v4/README.md) |

## 先看这些视频

- [正式基线：Original / V3 AA124](v3/v3_baseline/videos/Original_vs_V3_Baseline_AA_124.mp4)
- [滑窗：Global A继续 / D切换158](v3/sliding_windows/global/videos/G1_A_vs_D_158.mp4)
- [位置对照：Global / Local D158](v3/sliding_windows/local/videos/G1_vs_L1_D_158.mp4)
- [普通减步：30步 / 全程8步 + Global滑窗158](v3/fm2anyflow/fm8/videos/A_SWG30_vs_FM8_SWG8_158.mp4)
- [AnyFlow：普通FM-8NFE / AF-8NFE AD73](v3/fm2anyflow/anyflow/videos/AD_FM8_vs_AF8_continuation_73.mp4)
- [V4负结果：FM-8NFE / AF-8NFE / DMD8 AA56](v4/dmd/videos/AA_FM8_AF8_DMD8_cycle8_collapse_56.mp4)

[按研究方向播放全部视频](index.html) · [缩写表：FM-8NFE到底是什么意思](GLOSSARY.md) · [研究路线](roadmap.md) · [当前下一步](02_next_steps/README.md)

## 数据准备放在哪里

[模型生成的候选监督](v3/fm2anyflow/data_preparation/teacher_generated_targets/README.md)是原`v3_teacher_targets`：用较慢V3 FM-30NFE生成可能的训练答案，再检查动作是否正确。四图中s2/s3的D反向未证实，不能全部直接作为合格监督。

[真实数据与VAE重建](v3/fm2anyflow/data_preparation/real_data/README.md)展示原始录屏和压缩重建，不是生成能力展示。EXP-017固定布局仅做CPU验证，没有模型生成视频。

## 目录规则

`report/`存易读说明和可直接打开的代表性MP4；`experiments/`保存任务原始证据；`mainline/`记录版本定义并链接视频报告。报告叶子目录的`video_manifest.json`记录复制视频的来源与SHA-256。分组页只负责导航，不重复放视频。

V4目前只有研究尝试，未获得生成能力验收。旧实验ID与源码路径保留历史名称以支持复现，最新分类见[迁移记录](../archive/report_taxonomy_20261011/README.md)。
