# V0 · Original H3-World (Bidirectional Baseline)

## 1. Version Name / Research Objective

保留原始 H3-World 的动作和画面正控，作为后续协议的 reference。

## 2. Parent Version / Baseline

公开 MiniMax-H3 底座 + released H3-World action LoRA；不需要重训 bidirectional pretrained model。

## 3. Main Changes

完整视频联合去噪；原生 Single-Egress：Vᵢ直接读Aᵢ，Aᵢ只直接读对应video；跨时间动作可通过video hidden传播。

## 4. Model and Inference Configuration

主片：832×480，124 RGB / 24fps，seed13，30 full-horizon steps，flow shift2.22，Single I0。50-step旧片另存为历史reference，不能与主片合并统计。

## 5. Representative Videos

- [A/D原始方向正控](videos/V0_AD_reference.mp4)
- [W/S/A/D原始总览](videos/V0_WSAD_reference.mp4)
- [Original 10秒完整参考](videos/original_W_10s_243f.mp4)
- [Original 20秒完整参考](videos/original_W_20s_481f.mp4)

## 6. Quantitative Results

124f A/D horizontal flow：+1.0767 / −1.6019，separation 2.6786。39f旧teacher的2.023是另一长度/协议，不拼成同一学习曲线。

## 7. What Was Improved

提供动作方向及人物结构正控，也保留了10s/20s原始生成参考，撤回“Original天然不能生成长片”的旧说法。

## 8. What Still Failed

非GT；原始长片也有场景几何变形。此处没有因果交互、历史KV复用或首屏低延迟证明。

## 9. Lessons Learned

先固定正控，再区分attention、时间、图像条件和采样预算的影响；不能仅因seed相同就宣称噪声完全相同。

## 10. Source Code / Checkpoint / Original Experiment References

H3-World/outputs/2026-10-01-20/action_{A,D,W,S}_baseline30_124/；released LoRA SHA256 ddd9187b920b1e52c2d090f4e264fd83d8d433efc2a5b159e58883aeaf96e526。

[核心代码说明](code/README.md) · [代码SHA与源路径](code/SOURCE_MANIFEST.json) · [本版来源清单](PROVENANCE.json) · [返回汇报导航](../README.md)

指标均为历史记录；flow是运动proxy，MAD是活动量/连续性描述，不是视频质量评分。Original生成视频不是GT。
