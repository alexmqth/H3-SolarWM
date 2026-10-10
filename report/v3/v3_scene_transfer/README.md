# V3 两个固定场景的短程迁移 — EXP-011

**FM30与全程普通FM8均完成工业/村落首39帧和各自AA/AD56帧，有限可行PASS，quality PARTIAL。** 零新增训练，原始Single I0、native timestep、strict causal + persistent raw KV、Global位置与动作协议保持。FM30是causal V3。

工业场景两方法人物和建筑可用、动作分叉可辨。村落FM30有树木遮挡后重现，FM8仍有动作响应，但人物/植被重影和颗粒更明显。有限可行不代表质量相同。固定两初图/单seed/56帧不能推广为广泛或长时泛化。

G1首窗为匹配输入的步数比较；G2各方法使用自身生成C1，属于闭环比较。AA/AD在同方法内使用相同C1与噪声。

| 场景 | 持续A：FM30 / FM8 | A→D：FM30 / FM8 | FM30动作反事实 | FM8动作反事实 |
| --- | --- | --- | --- | --- |
| 工业 | [AA56](../../../experiments/EXP-011_v3_scene_transfer/artifacts/G2/comparisons/industrial_AA_FM30_vs_FM8_56.mp4) | [AD56](../../../experiments/EXP-011_v3_scene_transfer/artifacts/G2/comparisons/industrial_AD_FM30_vs_FM8_56.mp4) | [AA/AD](../../../experiments/EXP-011_v3_scene_transfer/artifacts/G2/comparisons/industrial_FM30_AA_vs_AD_56.mp4) | [AA/AD](../../../experiments/EXP-011_v3_scene_transfer/artifacts/G2/comparisons/industrial_FM8_AA_vs_AD_56.mp4) |
| 村落 | [AA56](../../../experiments/EXP-011_v3_scene_transfer/artifacts/G2/comparisons/village_AA_FM30_vs_FM8_56.mp4) | [AD56](../../../experiments/EXP-011_v3_scene_transfer/artifacts/G2/comparisons/village_AD_FM30_vs_FM8_56.mp4) | [AA/AD](../../../experiments/EXP-011_v3_scene_transfer/artifacts/G2/comparisons/village_FM30_AA_vs_AD_56.mp4) | [AA/AD](../../../experiments/EXP-011_v3_scene_transfer/artifacts/G2/comparisons/village_FM8_AA_vs_AD_56.mp4) |

总232forward/12decode，另6text/2image encode，含失败启动共0.514521GPUh。新片前39RGB不变，四套自身C1 rawKV已实际独立核查。G1采样部分约3.30×/3.73×加速；不代表完整E2E加速。

[Judge最终验收](../../../experiments/EXP-011_v3_scene_transfer/judge/FINAL_REVIEW.md) · [实验及Worker证据入口](../../../experiments/EXP-011_v3_scene_transfer/README.md) · [V3家族](../README.md)
