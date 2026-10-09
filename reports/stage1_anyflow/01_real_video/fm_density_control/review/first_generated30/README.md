# 第一自然场景：generated-history 30/chunk全部39帧评审

2026-10-09。片段`118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140`；新shift2.22与旧shift12均普通FM48，相同首帧/联合动作/prompt/video及audio noise。先前GT-history诊断之外，本次历史与后续RGB anchor都来自各自生成结果。

查看了新旧全部39帧接触表和新候选放大18–38帧，属于静态逐帧评审。人物仍可辨，末段没有完全分解；人物动作与场景位移偏弱，背景结构/植被仍变化，未见相比旧FM48的明确质变。不能把人物保住或像素活动降低说成恢复动作控制。这个自然片段同时包含W+A与camera pan，不是纯A/D正控。

| 方法 | Frame gray MAD | Boundary RGB MAD | Mean horizontal flow |
|---|---:|---:|---:|
| Original30 full-seq | 19.052133 | 18.753836 | +2.288792 |
| 旧FM48 generated30/chunk | 7.758413 | 9.514335 | −0.142475 |
| 新FM48 generated30/chunk | 6.940786 | 7.701437 | −0.217875 |

新候选载入后519.82s，端到端524.19s；GPU allocated26002.32MiB，CPU KV6484.13MiB；90 noisy forwards＋3 clean commits。共享主机单次、cached conditioning，不能据此声称提速。边界MAD下降伴随整体运动量下降，不独立支持画质提升。

[候选完整视频](video.mp4) · [评测收据](evaluation.json) · [新全部帧](new_all39.jpg) · [旧全部帧](old_all39.jpg) · [放大18–38](new_18_38.jpg) · [输入/视频hash](sources.json)。本场景不能代表第二场景或8步/纯A/D结果，完整Stage1尚未通过。
