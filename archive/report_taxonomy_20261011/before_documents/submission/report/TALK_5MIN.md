# 五分钟汇报：V3 Baseline与Sliding Window候选

## 0:00–0:50：问题从哪里来

Original H3具备动作与视觉基础。V1迁入chunk-causal attention和persistent KV，工程可以运行，但出现视觉与动作退化。展示[Original / V1对照](00_comparison_gallery/V1_vs_original.mp4)。

## 0:50–2:10：三条并行方案

[V2路线](v2/README.md)与[V3正式参考](v3/README.md)。V2a通过RGB Anchor和adaptation修复视觉条件，124帧基本结构可用，动作失败；V2b保留Same-σ历史/当前联合双向去噪，持续A/D124帧动作与基本结构可用，计算昂贵且无persistent KV；V3 Baseline使用原生条件与冻结历史KV，实现严格因果、动作与基本结构的124帧可行性。

这些路线都在解决V1退化问题，没有顺次权重继承关系。V3 Baseline使用Original H3和released Action LoRA，保持已验收协议与结果。

## 2:10–3:30：看已有能力与缺陷

展示[V2b与Original持续A/D124帧](v2/v2b_same_sigma_local_bidir/videos/Original_vs_V2b_AD_overview_248.mp4)，再展示[V2b / V3 Baseline AA124](v3/v3_baseline/videos/V2b_vs_V3_Baseline_AA_124.mp4)。V3 Baseline真实复用KV、无需每步重算祖先，但AA约79–86帧有明显形变后恢复，连续性PARTIAL。V2a的[20秒失败片](v2/v2a_rgb_anchor/videos/V2a_long20s_failure.mp4)保留完整后段。跨路线比较包含多项协议变化，不能作单因素归因或公平E2E速度比。

## 3:30–4:25：最便宜的少步验证

[EXP-004：V3 Baseline 30步 / 8步续写](v3/v3_baseline/8step_continuation/README.md)。原权重普通FM、不新增训练，两路径到73帧，动作与基本结构保留，AA拖影明显。34forward、4VAE、0.129449 GPU小时。首39帧仍复用30步结果，不能声称全程8步或AnyFlow成功。

## 4:25–5:00：下一步与停止条件

下一项高ROI问题是超过六块后真实淘汰历史KV，能否维持动作与基本结构。先复现首次淘汰前的V3，再做SW-G第7/8块；SW-L保持同窗口，仅改变video位置，独立决定是否验证。当前只有CPU工作授权，GPU提案需另批。明显失败则有限预算收口，不靠改prefix/time或追加训练救结果。

目录按`report/v2/`与`report/v3/`分组。V3下分别为Baseline、SW-G、SW-L；FM8和AnyFlow是后续独立计划。
