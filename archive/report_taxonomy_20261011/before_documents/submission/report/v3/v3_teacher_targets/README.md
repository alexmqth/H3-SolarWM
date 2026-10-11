# V3 FM30 四初图教师候选：EXP-014

这一组视频检查的是：冻结的 **V3 Original Feasibility Baseline** 在四张确定的训练初图上，能否产生可用于后续研究的局部 A/D 反事实目标。它不是新 checkpoint，也没有新增训练。每图先以 A 生成 C1 39帧，再从**同一份自生成 C1、同一份 clean raw KV 与同一 C2 噪声**分叉：AA 继续 A，AD 切换 D。两块共56 RGB帧、30步/块、native Single I0、strict chunk-causal、Global RoPE、persistent CPU raw KV；动作是合成反事实，不是录屏的 GT 标签。

| 场景 | AA/AD 并排视频 | C2 中央水平光流和 AA / AD | 当前可用边界 |
|---|---|---:|---|
| `s0` 山路 | [56帧](videos/s0_AA_vs_AD_56.mp4) | +42.15 / −27.46 | 有限可用的动作分叉；人物与环境可辨，局部软化。 |
| `s1` 草地道路 | [56帧](videos/s1_AA_vs_AD_56.mp4) | +3.65 / −45.31 | D 反向较清楚；持续 A 净位移弱且途中变号，质量PARTIAL。 |
| `s2` 工业道路 | [56帧](videos/s2_AA_vs_AD_56.mp4) | +68.38 / +50.07 | 人物结构可辨，但 D 反向未证实，不宜作为动作正确的AD监督。 |
| `s3` 暗色草地 | [56帧](videos/s3_AA_vs_AD_56.mp4) | +66.18 / +58.66 | 两分支几乎同向、像素差仅约3.19；D反向未证实，暗部和草有遮挡。 |

这四个并排片均为已保存真实H3输出的副本，832×480原片并排为1664×552、24FPS、56帧，已完整解码验证。`s1/s3` 的 AD 是在原尝试意外中断后，**只从已保存 C1/KV/噪声重新计算缺失的30步**所得；旧半途日志和账本保留。对应源视频和 SHA 在[视频来源清单](video_manifest.json)，[完整实验入口](../../../experiments/EXP-014_v3_multiscene_teacher/README.md)保存原片、端点、逐调用账本、恢复审计与资源成本。

Judge 已对实验作出 **ACCEPT WITH LIMITS** 的[正式验收](../../../experiments/EXP-014_v3_multiscene_teacher/judge/FINAL_REVIEW.md)：局部生成和协议链路在四张训练初图上成立，**四图动作方向全部正确的监督未成立**。水平光流只是场景运动辅助量，不能单独代表人物动作。[正式逐场用途标签](../../../experiments/EXP-014_v3_multiscene_teacher/judge/TARGET_USE_ASSESSMENT.json)把`s2/s3`排除出“D反向正确”监督。不要将此组教师候选误写成已完成的多场景 AnyFlow 或 Stage2/DMD；两者均未在本任务训练。

[V3家族导航](../README.md) · [V3正式124帧基线](../v3_baseline/README.md) · [EXP-014恢复GPU报告](../../../experiments/EXP-014_v3_multiscene_teacher/RECOVERY_V2_GPU_REPORT.md)
