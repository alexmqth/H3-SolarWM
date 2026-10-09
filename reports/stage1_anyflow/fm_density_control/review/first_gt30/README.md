# 首条新FM48 GT-history30：未见明确画质跃升

2026-10-09 01:50。第一验证场景，完整39帧；新旧FM48使用同一首帧、prompt/action、video/audio noise和GT/RGB/CPU-KV条件，仅训练density不同。Original列为完整序列原始推理参照，GT-history列每块使用真实历史，不能混同自由rollout。

**静态逐帧结论：人物在39帧中持续可辨，没有在本片中看到完全分解；与旧FM48相比没有明确质变。草木、楼体、视角/比例偏离仍在，17/34帧GT-history边界重置仍明显。** 这不是实时播放评价，不证明自主生成历史稳定或A/D方向恢复。

| 方法 | frame gray MAD | boundary RGB MAD | mean horizontal flow |
|---|---:|---:|---:|
| Original30 full-sequence | 19.052133 | 18.753836 | 2.288792 |
| FM48 shift12 GT30/chunk | 15.978593 | 46.474616 | 0.561248 |
| FM48 shift2.22 GT30/chunk | 14.917393 | 46.338868 | 0.565374 |

MAD较低可表示活动量变少，不能等同于画质提升；此处真实联合按键/镜头动作的flow不是纯A/D正确率。
新分支记录推理after-load452.24s，allocated GPU26002.10MiB、CPU KV6484.13MiB，90 noisy forwards＋3 commits。单次共享主机、不含重新计算conditioning，不声称公平加速或分解weights/activations。

[完整39帧视频](candidate_gt30.mp4) · [全部39帧](candidate_all39.jpg) · [18–38帧大图](candidate_late18_38.jpg) · [GT/Original/旧/新对应帧](matched.jpg) · [检查收据](contact_receipt.json) · [指标](metrics.json)

4个源视频均完整解码为39帧、24fps、H264/YUV420P；视频hash和输入fingerprint匹配结果保留。这里只审阅已完成的单个片段，第二GT场景、generated30/8、停车场A/D与geometry仍由原队列执行。没有替换meeting，也不作为新主Demo。
