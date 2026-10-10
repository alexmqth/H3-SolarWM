# V3-AF：新student的有限少步续写

**EXP-007已验收：累计32次finite-map训练后，8NFE AA/AD73帧续写有限可行性通过；画质/连续性PARTIAL，尚无整体优于普通FM8的证据。** 正式V3 Baseline冻结。

AF0/AF1建立新的target-time gate .25和last8 rank8 QKV，初始化对角输出与V3逐值一致。AF2新增31updates/527forward/124backward/1.191072GPU小时，累计32updates。每次更新后重建自身clean KV；没有加载旧AnyFlow训练权重。

AF3使用共同FM8首39帧，各模型用自身权重构建KV。C2同clean历史、动作、噪声和8NFE；C3各自生成历史。A/D方向C2为+0.650/−0.697，C3为+1.006/−0.149。人物与场景可辨，但边界跳变和肢体残影保留。只验收单scene/seed13、73帧continuation，不是从首窗全程AF或长时滑窗验证。

AF3实际35forward/4VAE/0update/473.000176秒=0.131389GPU小时，peak26.33646GiB；包含首次模型forward前的导入失败。训练C1来自FM30，评估C1来自FM8，这一分布变化保留；不宣称纯训练单因素效应或公平E2E速度提升。

[AA原片](../../../experiments/EXP-007_v3_anyflow/artifacts/af3_raw/AA/rollout_73.mp4) · [AD原片](../../../experiments/EXP-007_v3_anyflow/artifacts/af3_raw/AD/rollout_73.mp4) · [AA与FM8并排](../../../experiments/EXP-007_v3_anyflow/artifacts/af3_videos/AA_FM8_vs_AF8_continuation_73.mp4) · [AD与FM8并排](../../../experiments/EXP-007_v3_anyflow/artifacts/af3_videos/AD_FM8_vs_AF8_continuation_73.mp4)

[完整实验](../../../experiments/EXP-007_v3_anyflow/README.md) · [AF2训练审核](../../../experiments/EXP-007_v3_anyflow/judge/AF2_REVIEW.md) · [AF3视频审核](../../../experiments/EXP-007_v3_anyflow/judge/AF3_REVIEW.md) · [普通FM8参考](../v3_fm8/README.md)

下一研究是独立DMD工程pilot；本轮没有DMD结果，不为普通视觉缺陷继续AnyFlow调参。

## 后续4NFE负结果

同一AF2 step32在EXP-009的4NFE续写中，AA仍有动作但人体透明加重，AD C3持续人物分解、方向近消失，该分支生成FAIL。普通FM4相同条件下AA/AD73仍有限可用；未显示AnyFlow4整体收益，当前4NFE配置停止，不追加步数扫描或训练。[4NFE完整结论](../v3_fm4_af4/README.md)。这不修改本页8NFE有限可行性结论。
