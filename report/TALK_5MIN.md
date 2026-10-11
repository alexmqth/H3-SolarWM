# 五分钟汇报：V3 Baseline与Sliding Window候选

## 0:00–0:50：问题从哪里来

Original H3具备动作与视觉基础。V1迁入chunk-causal attention和persistent KV，工程可以运行，但出现视觉与动作退化。展示[Original / V1对照](00_comparison_gallery/V1_vs_original.mp4)。

## 0:50–2:10：三条并行方案

[V2路线](v2/README.md)与[V3正式参考](v3/README.md)。V2a通过RGB Anchor和adaptation修复视觉条件，124帧基本结构可用，动作失败；V2b保留Same-σ历史/当前联合双向去噪，持续A/D124帧动作与基本结构可用，计算昂贵且无persistent KV；V3 Baseline使用原生条件与冻结历史KV，实现严格因果、动作与基本结构的124帧可行性。

这些路线都在解决V1退化问题，没有顺次权重继承关系。V3 Baseline使用Original H3和released Action LoRA，保持已验收协议与结果。

## 2:10–3:30：看已有能力与缺陷

展示[V2b与Original持续A/D124帧](v2/v2b_same_sigma_local_bidir/videos/Original_vs_V2b_AD_overview_248.mp4)，再展示[V2b / V3 Baseline AA124](v3/v3_baseline/videos/V2b_vs_V3_Baseline_AA_124.mp4)。V3 Baseline真实复用KV、无需每步重算祖先，但AA约79–86帧有明显形变后恢复，连续性PARTIAL。V2a的[20秒失败片](v2/v2a_rgb_anchor/videos/V2a_long20s_failure.mp4)保留完整后段。跨路线比较包含多项协议变化，不能作单因素归因或公平E2E速度比。

## 3:30–4:25：两条V3研究路线与V4结果

[Sliding Window](v3/sliding_windows/README.md)解决历史video KV增长：Global最近5祖先淘汰后158帧有限可行；Local未显示联合收益，当前无训练方向归档。

[FM → AnyFlow](v3/fm2anyflow/README.md)解决采样成本：FM-8NFE（每块8步普通流匹配采样）已测新首窗与滑窗158帧；AF-8NFE（AnyFlow每块8次求值）停车场有限可行，但迁移无整体优势，AF-4NFE当前配置失败。[V4 SGF / DMD](v4/README.md)训练工程通过，最终新帧彩噪，当前配置停止。

## 4:25–5:00：下一步与停止条件

[教师候选与真实数据](v3/fm2anyflow/data_preparation/README.md)帮助检查后续训练的监督和输入。教师四图不能全部作为动作正确标签；真实数据/VAE与固定布局CPU候选已分别验收。下一建议是实际模型回归，再做真实混合动作的有限局部续写，见[草案](02_next_steps/README.md)。当前无新GPU任务。

普通画质缺陷按可行性尺度记录PARTIAL；持续结构崩坏时停止对应配置。V3按Sliding Window与FM → AnyFlow分组，DMD归V4；[当前导航](README.md)可直接打开视频与名称解释。
