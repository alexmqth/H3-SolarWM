# 五分钟汇报：V2三条并行修复路线

## 0:00–0:50：问题从哪里来

Original H3具备动作与视觉基础。V1迁入chunk-causal attention和persistent KV，工程可以运行，但出现视觉与动作退化。展示[Original / V1对照](00_comparison_gallery/V1_vs_original.mp4)。

## 0:50–2:10：三条并行方案

[V2路线总览](v2/README.md)。V2a通过RGB Anchor和adaptation修复视觉条件，124帧基本结构可用，动作失败；V2b保留Same-σ历史/当前联合双向去噪，持续A/D124帧动作与基本结构可用，计算昂贵且无persistent KV；V2c使用原生条件与冻结历史KV，实现严格因果、动作与基本结构的124帧可行性。

三者都在解决V1退化问题。字母表示方法，不表示V2a→V2b→V2c的权重继承。V2c是原来称为V3 Efficient Causal的路线，本次调整研究分类，已有实验结论不变。

## 2:10–3:30：看已有能力与缺陷

展示[V2b与Original持续A/D124帧](v2/v2b_same_sigma_local_bidir/videos/Original_vs_V2b_AD_overview_248.mp4)，再展示[V2b / V2c AA124](v2/v2c_strict_causal_kv/videos/V2b_vs_V2c_AA_124.mp4)。V2c真实复用KV、无需每步重算祖先，但AA约79–86帧有明显形变后恢复，连续性PARTIAL。V2a的[20秒失败片](v2/v2a_rgb_anchor/videos/V2a_long20s_failure.mp4)保留完整后段。跨路线比较包含多项协议变化，不能作单因素归因或公平E2E速度比。

## 3:30–4:25：最便宜的少步验证

[EXP-004：V2c 30步 / 8步续写](v2/v2c_strict_causal_kv/8step_continuation/README.md)。原权重普通FM、不新增训练，两路径到73帧，动作与基本结构保留，AA拖影明显。34forward、4VAE、0.129449 GPU小时。首39帧仍复用30步结果，不能声称全程8步或AnyFlow成功。

## 4:25–5:00：下一步与停止条件

下一项高ROI问题是首窗也使用8步后，能否从头启动并沿自己的历史继续生成。先短片判断，有效再有限延伸；明显失败则收口，不扫描大量参数或默认扩大训练。AnyFlow/DMD按后续证据决定，未来V3研究定义尚未发布。

目录统一按report/v2/v2a、v2b、v2c方法分组，后续有变体的版本沿用同一规则。
