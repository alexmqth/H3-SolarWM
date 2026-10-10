# V3-AF：新student的有限少步续写

**EXP-007已验收：累计32次finite-map训练后，8NFE AA/AD73帧续写有限可行性通过；画质/连续性PARTIAL，尚无整体优于普通FM8的证据。** 正式V3 Baseline冻结。

AF0/AF1建立新的target-time gate .25和last8 rank8 QKV，初始化对角输出与V3逐值一致。AF2新增31updates/527forward/124backward/1.191072GPU小时，累计32updates。每次更新后重建自身clean KV；没有加载旧AnyFlow训练权重。

AF3使用共同FM8首39帧，各模型用自身权重构建KV。C2同clean历史、动作、噪声和8NFE；C3各自生成历史。A/D方向C2为+0.650/−0.697，C3为+1.006/−0.149。人物与场景可辨，但边界跳变和肢体残影保留。只验收单scene/seed13、73帧continuation，不是从首窗全程AF或长时滑窗验证。

AF3实际35forward/4VAE/0update/473.000176秒=0.131389GPU小时，peak26.33646GiB；包含首次模型forward前的导入失败。训练C1来自FM30，评估C1来自FM8，这一分布变化保留；不宣称纯训练单因素效应或公平E2E速度提升。

[AA原片](../../../experiments/EXP-007_v3_anyflow/artifacts/af3_raw/AA/rollout_73.mp4) · [AD原片](../../../experiments/EXP-007_v3_anyflow/artifacts/af3_raw/AD/rollout_73.mp4) · [AA与FM8并排](../../../experiments/EXP-007_v3_anyflow/artifacts/af3_videos/AA_FM8_vs_AF8_continuation_73.mp4) · [AD与FM8并排](../../../experiments/EXP-007_v3_anyflow/artifacts/af3_videos/AD_FM8_vs_AF8_continuation_73.mp4)

[完整实验](../../../experiments/EXP-007_v3_anyflow/README.md) · [AF2训练审核](../../../experiments/EXP-007_v3_anyflow/judge/AF2_REVIEW.md) · [AF3视频审核](../../../experiments/EXP-007_v3_anyflow/judge/AF3_REVIEW.md) · [普通FM8参考](../v3_fm8/README.md)

后续DMD工程与视频验证已在EXP-008独立完成，当前cycle8生成失败已归档；本页保留EXP-007的历史结论。

## 后续4NFE负结果

同一AF2 step32在EXP-009的4NFE续写中，AA仍有动作但人体透明加重，AD C3持续人物分解、方向近消失，该分支生成FAIL。普通FM4相同条件下AA/AD73仍有限可用；未显示AnyFlow4整体收益，当前4NFE配置停止，不追加步数扫描或训练。[4NFE完整结论](../v3_fm4_af4/README.md)。这不修改本页8NFE有限可行性结论。

## 两个固定其他场景的AF8迁移（EXP-012）

**当前step32未通过联合迁移验证，无已证实的FM8替代收益。** 工业画面可用，但AD C2反向响应不清楚，切换PARTIAL（辅助flow+11.07，对照FM8−38.27）；村落AD仍可用，AA后半块人物/前景出现大片分叉透明残影、结构FAIL，明显重于FM8。停车场旧8NFE有限可行结论保持，不据两个场景否定全部AnyFlow。

同EXP-011 FM8首39 latent/RGB，各模型自己构建raw KV，C2匹配动作/噪声/native8网格；不是AF从首窗生成。实际34forward/4decode/0训练，354.403秒=.098445GPUh，峰26.230GiB。两套真实50层AF KV均与FM8内容不同但位置相同，历史RGB不变。

[工业AA](../../../experiments/EXP-012_v3_af8_scene_transfer/artifacts/comparisons/industrial_AA_FM8_vs_AF8_56.mp4) · [工业AD](../../../experiments/EXP-012_v3_af8_scene_transfer/artifacts/comparisons/industrial_AD_FM8_vs_AF8_56.mp4) · [村落AA结构失败](../../../experiments/EXP-012_v3_af8_scene_transfer/artifacts/comparisons/village_AA_FM8_vs_AF8_56.mp4) · [村落AD](../../../experiments/EXP-012_v3_af8_scene_transfer/artifacts/comparisons/village_AD_FM8_vs_AF8_56.mp4) · [Judge最终审核](../../../experiments/EXP-012_v3_af8_scene_transfer/judge/FINAL_REVIEW.md)。

本轮收口，无C3/调参/追加训练。下一步先核查多场景数据、episode隔离与native Single I0教师目标方案，再决定新的训练任务。
