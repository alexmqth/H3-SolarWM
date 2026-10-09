# FM48首条完整视频：第一验证场景，GT history，30 steps/chunk

**这里只完成第一场景的GT-history检查，不能作为完整Stage1或generated-history验收。** 输入与Original/causal0的conditioning/noise记录一致。39帧、832×480、24fps、H264/yuv420p、faststart，完整解码通过。

[完整原视频](video.mp4) · [四行对应帧](matched.jpg) · [训练后全部39帧](Causal48_GT30_all39.jpg) · [训练前全部39帧](Causal0_GT30_all39.jpg)

| 方法 | 帧间灰度MAD | 边界RGB MAD | 水平光流均值 | 载入后推理秒数 |
|---|---:|---:|---:|---:|
| Original30 full-sequence | 19.0521 | 18.7538 | 2.2888 | 227.67 |
| Causal0 GT30/chunk | 17.1435 | 47.2777 | 0.8856 | 533.97 |
| Causal48 GT30/chunk | 15.9786 | 46.4746 | 0.5612 | 522.29 |

逐张检查训练前后全部39帧contact sheet和Original/真实GT的0/8/16/24/30/38对应帧，非实时播放。FM48人物全程可辨，没有在这条GT-history片段看到此前第二场景generated-history的严重人物分解。但树木/建筑细节和相对位置仍漂移，17/34帧附近明显重置，未见明确的视觉质变。训练前这条GT-history的人物也保留，不能将这一点声称为FM48新获得的能力。

每个chunk都重新使用真实GT前缀，前一预测endpoint与下一段GT条件不匹配会产生拼接跳变；这种oracle诊断不能与自主rollout失败混同。整段水平光流降低只说明这条联合动作/镜头片段的图像水平运动变弱，不是纯A/D准确率。MAD降低也可能来自运动降低或模糊，不作为画质提高。

Causal每条90次noisy forwards＋3次clean commits，CPU KV6484.13MiB、GPU allocated peak26002.10MiB；Original整段30次。时间为共享主机单次测量，不能宣称训练带来加速。完整动作对照和第二验证场景仍在队列中。

[原始评测记录](evaluation.json) · [完整解码检查](video_validation.json) · [输入检查](preparation.json)
