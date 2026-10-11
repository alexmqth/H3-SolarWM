# V3 场景迁移：FM-30NFE / FM-8NFE 对照

## 名称与目的

**名称解释：** FM = Flow Matching（流匹配）；30和8是每块采样步数。scene transfer表示换到其他固定场景测试，不表示本轮训练了迁移模型。

**本次目的：** 检查停车场观察到的普通减步可行性，在工业和村落初图上能否保留；用各自生成的首窗续写，观察动作、结构与质量损失。

[其他缩写与计数](../../../GLOSSARY.md) · [汇报总览](../../../README.md)

## 实验结果与证据

**FM-30NFE与全程普通FM-8NFE均完成工业/村落首39帧和各自AA/AD56帧，有限可行PASS，quality PARTIAL。** 零新增训练，原始Single I0、native timestep、strict causal + persistent raw KV、Global位置与动作协议保持。FM-30NFE是causal V3。

工业场景两方法人物和建筑可用、动作分叉可辨。村落FM-30NFE有树木遮挡后重现，FM-8NFE仍有动作响应，但人物/植被重影和颗粒更明显。有限可行不代表质量相同。固定两初图/单seed/56帧不能推广为广泛或长时泛化。

G1首窗为匹配输入的步数比较；G2各方法使用自身生成C1，属于闭环比较。AA/AD在同方法内使用相同C1与噪声。

| 场景 | 持续A：FM-30NFE / FM-8NFE | A→D：FM-30NFE / FM-8NFE | FM-30NFE动作反事实 | FM-8NFE动作反事实 |
| --- | --- | --- | --- | --- |
| 工业 | [AA56](videos/industrial_AA_FM30_vs_FM8_56.mp4) | [AD56](videos/industrial_AD_FM30_vs_FM8_56.mp4) | [AA/AD](videos/industrial_FM30_AA_vs_AD_56.mp4) | [AA/AD](videos/industrial_FM8_AA_vs_AD_56.mp4) |
| 村落 | [AA56](videos/village_AA_FM30_vs_FM8_56.mp4) | [AD56](videos/village_AD_FM30_vs_FM8_56.mp4) | [AA/AD](videos/village_FM30_AA_vs_AD_56.mp4) | [AA/AD](videos/village_FM8_AA_vs_AD_56.mp4) |

总232forward/12decode，另6text/2image encode，含失败启动共0.514521GPUh。新片前39RGB不变，四套自身C1 rawKV已实际独立核查。G1采样部分约3.30×/3.73×加速；不代表完整E2E加速。

[Judge最终验收](../../../../experiments/EXP-011_v3_scene_transfer/judge/FINAL_REVIEW.md) · [实验及Worker证据入口](../../../../experiments/EXP-011_v3_scene_transfer/README.md) · [V3家族](../../README.md)

## 视频来源

本目录 `videos/` 为已完成实验视频的原字节副本，可直接打开；来源、大小和 SHA-256 见[视频清单](video_manifest.json)。
