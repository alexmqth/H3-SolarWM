# V2a · RGB-Anchor Causal H3-World

## 1. Version Name / Research Objective

修正latent-only条件与RGB视觉训练条件不一致的问题，改善V1家族的generated-history视觉退化。

## 2. Parent Version / Baseline

针对V1暴露的生成历史/条件不匹配问题的一条并列修复路线；使用RGB visual QKV与已有fixed-mix action residual。与V2b没有顺承/权重继承关系。

## 3. Main Changes

生成prefix latent→VAE decode→取末RGB→process_image=True重编码；RGB-consistent dual anchor。联合tail16 QKV普通teacher replay、Original endpoint和boundary监督；causal action prefix + feedback。

## 4. Model and Inference Configuration

visual_online_rgb_tail16_endpoint_ad2；尾16 QKV两次online更新，冻结fixed-mix action residual；124f/24fps，seed13，8/chunk×8，CPU KV。没有AnyFlow或DMD student更新。

## 5. Representative Videos

- [V2a vs V2b：并列能力比较，非单变量消融](../../report/V2a_rgb_anchor/videos/V2a_vs_V2b.mp4)
- [Original vs V2a，A/D124f](../../report/V2a_rgb_anchor/videos/V2a_vs_original.mp4)
- [V1→V2a，联合协议对比](../../report/V2a_rgb_anchor/videos/V2a_vs_V1.mp4)
- [同checkpoint20秒完整失败证据](../../report/V2a_rgb_anchor/videos/V2a_long20s_failure.mp4)
- [V2a A原片](../../report/V2a_rgb_anchor/videos/V2a_A_full.mp4)
- [V2a D原片](../../report/V2a_rgb_anchor/videos/V2a_D_full.mp4)

## 6. Quantitative Results

124f A=−0.7841、D=−1.0075，separation0.2233；A/D E2E699.5/715.9s，CPU KV13.19GiB。新主表RGB MAD3.1855/3.0356、boundary3.7871/3.8874；与旧报告的另一MAD实现分开，未改写原值。

## 7. What Was Improved

**Longer-horizon Visual Stability Demonstrated — 仅124f的相对视觉证据，非无限长稳定。**

所选124f中人物和停车场结构相对视觉崩坏版本保持得更好。改善属于RGB协议、visual adapter和监督等联合方案，不能全部归功于anchor。

## 8. What Still Failed

A/D仍同为负，动作验收失败；10s后段退化，20s约10秒起雾化，15秒后人物/场景难辨。保留完整失败视频。

## 9. Lessons Learned

视觉稳定与动作保真是不同验收轴。对齐image-conditioning可修复一部分结构问题，不能替代动作信息流恢复。

## 10. Source Code / Checkpoint / Original Experiment References

checkpoints/visual_rgb_tail16/{causal_adapter.pt,action_adapter.pt}；原实验H3-World/outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/。adapter未复制进汇报包。

[核心代码说明](../../report/V2a_rgb_anchor/code/README.md) · [代码SHA与源路径](../../report/V2a_rgb_anchor/code/SOURCE_MANIFEST.json) · [本版来源清单](../../report/V2a_rgb_anchor/PROVENANCE.json) · [返回汇报导航](../../report/README.md)

指标均为历史记录；flow是运动proxy，MAD是活动量/连续性描述，不是视频质量评分。Original生成视频不是GT。

## 完整证据与边界

[原实验/配置/视频对应manifest](manifest.json) · [整理前冻结的版本映射](../../archive/reorganization_20261010/version_map.json)

[原视觉修复报告](../../reports/visual_drift_repair/VISUAL_DRIFT_REPAIR_REPORT.md) · [checkpoint来源](../../meeting/DEMO_PROVENANCE.md) · [长片完整负结果](../../meeting/long_horizon/README.md)
