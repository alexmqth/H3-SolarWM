# EXP-019：真实混合动作条件下的 V3 局部续写

四个固定训练 episodes 的真实前39帧用于 GT history，冻结 V3 原生条件与 strict causal raw KV 生成后17帧；分别用普通 FM30 和 FM8。视频共56帧，但**只有 RGB39–55 是新生成**。这项评测检查变长真实动作的输入布局及局部画面，未训练 AnyFlow/DMD，也不是从 I0 自由生成的完整56帧。

| Scene | 动作脚本概况 | GT / FM30 / FM8 | 局部判断 |
|---|---|---|---|
| 43866101 | 后退+左平移，相机缓慢右移 | [三列视频](../../../../../experiments/EXP-019_v3_mixed_action_validation/artifacts/G1_43866101/source_vs_fm30_vs_fm8.mp4) | 人物/山路可辨；语义 PARTIAL |
| 7199292c | 后退+左平移，相机跟随 | [三列视频](../../../../../experiments/EXP-019_v3_mixed_action_validation/artifacts/G2_7199292c/source_vs_fm30_vs_fm8.mp4) | 人物/草地可辨；无反事实动作验收 |
| 9dc2e588 | 后退+左平移，相机左转快慢变化 | [三列视频](../../../../../experiments/EXP-019_v3_mixed_action_validation/artifacts/G2_9dc2e588/source_vs_fm30_vs_fm8.mp4) | 人物可辨；FM8 背景结构变化明显 |
| b784d995 | 左平移→静止→前进 | [三列视频](../../../../../experiments/EXP-019_v3_mixed_action_validation/artifacts/G2_b784d995/source_vs_fm30_vs_fm8.mp4) | 人物可辨；切换控制未确证 |

原始输入、代码、完整指标、逐帧图、SHA 和审计在[实验目录](../../../../../experiments/EXP-019_v3_mixed_action_validation/README.md)，其中[G2 Worker 报告](../../../../../experiments/EXP-019_v3_mixed_action_validation/G2_WORKER_REPORT.md)明确说明各场景质量局限。整轮166次forward（含6次commit）、14次text、8次decode、0训练，总计2052.980 GPU秒。光流符号只是画面运动代理；训练 episodes 的 GT-history 局部结果不等于独立集泛化、反事实动作正确或自由生成长视频通过。Judge已accepted收口：局部可行性PASS，质量/动作PARTIAL，自由生成与泛化NOT_TESTED。

## 原版 H3 独立参考与最终审核

[Original全56帧与GT并排](../../../../../experiments/EXP-020_original_mixed_action_reference/artifacts/b784d995/gt_vs_original56.mp4)：同b784d995与seed13，30NFE，282.237秒，峰allocated26.049GiB。天空有放射伪影，动作切换未确证。Original全56帧均从I0生成，与上表GT39+生成17的条件不同，不能直接比较速度或作因果拓扑归因。

[EXP-019 Judge审核与耗时表](../../../../../experiments/EXP-019_v3_mixed_action_validation/judge/FINAL_REVIEW.md) · [EXP-020 Judge审核](../../../../../experiments/EXP-020_original_mixed_action_reference/judge/FINAL_REVIEW.md)。FM表示普通Flow Matching，NFE表示每个新chunk的网络函数求值次数；FM-8NFE没有新增AnyFlow训练。整轮166次DiT包括152生成采样、8回归探针和6commit。
