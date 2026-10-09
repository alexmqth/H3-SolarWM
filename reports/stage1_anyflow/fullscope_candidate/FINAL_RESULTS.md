# 全覆盖rank8 AnyFlow16：完整评测未通过

更新时间：2026-10-08T08:23:11.686457+08:00。4条39f A/D视频及数值评测全部完成。实际均载入43,237,376参数的全覆盖bank，输入首帧/prompt/action/seed/video+audio noise与对应原始H3逐张量一致，完整39帧/24fps/H264可解码。

- 4 steps/chunk：A=-1.198536、D=-1.242114、A-D=0.043578；A方向符号不符合原始H3参考，未通过。
- 8 steps/chunk：A=-1.305347、D=-1.496471、A-D=0.191124；A方向符号不符合原始H3参考，未通过。

| Action | Steps/chunk | Horizontal flow | E2E s | GPU allocated peak MiB | CPU KV MiB | Noisy+commit | Gray frame MAD | RGB boundary MAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 4 | -1.198536 | 228.22 | 38984.16 | 6484.13 | 12+3 | 4.0724 | 6.0693 |
| D | 4 | -1.242114 | 220.69 | 38984.16 | 6484.13 | 12+3 | 4.0485 | 6.4854 |
| A | 8 | -1.305347 | 267.08 | 38984.16 | 6484.13 | 24+3 | 3.8701 | 5.2849 |
| D | 8 | -1.496471 | 271.64 | 38984.16 | 6484.13 | 24+3 | 3.7232 | 4.9133 |

4-step的A/D后段均严重重影、雾化，人物与车库结构逐渐丢失，与native tail16没有明确修复。8-step的人物/场景较完整，但仍有模糊，动作方向未恢复，与native tail16对照无明确画质收益。A4、A8、D8已查看全部39帧，D4查看相同时间点对照（0/6/12/18/24/30/38）；所有原始完整视频都保留。分离度较native16略变不构成通过，尤其不能忽略A仍为负。

可播放诊断： [Original H3 / full-scope AnyFlow4 / full-scope AnyFlow8，A/D两行](fullscope16_AD_original_4step_8step_diagnostic.mp4)。这是失败诊断，不是新的最终展示视频；不替换meeting内已有材料。所有失败后段保留，视频标注步数、noisy/commit次数与记录耗时。

这里是单次、本机、CPU offload测量。原始teacher的offload预算没有记录，不能用这些时间作严格speedup；GPU allocated与nvidia-smi不同。MAD是运动活跃度，不能直接作为画质分数。16次更新仍很少，此结果不证明充分训练的AnyFlow不可行。

同容量native-FP32 full-scope FM16已接管GPU0；需待FM训练及A/D4/8评测结束再比较目标/条件/采样的效果。Stage1尚未达到验收，Stage2继续暂缓。另一项隔离的CPU候选正在验证clean-history梯度保留：它保持现有cache前向语义，不是官方融合两流算子，也尚无33B显存或画质结果。
