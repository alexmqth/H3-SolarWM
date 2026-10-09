# 横向比较画廊

全部来自已经保存的真实MP4；左旧右新，A/D分开再合并。V1_vs_V0与V1_vs_original为同义命名副本，其余不是伪造的视频。

| 视频 | 帧数 / 秒 | 用途 |
|---|---|---|
| [V0_AD_reference.mp4](V0_AD_reference.mp4) | 124 / 5.167 | V0 action reference: A vs D |
| [V0_WSAD_reference.mp4](V0_WSAD_reference.mp4) | 124 / 5.167 | V0 Original H3-World: W / S / A / D |
| [V1_A_vs_original.mp4](V1_A_vs_original.mp4) | 124 / 5.167 | Original H3-World vs V1 | action A |
| [V1_D_vs_original.mp4](V1_D_vs_original.mp4) | 124 / 5.167 | Original H3-World vs V1 | action D |
| [V1_vs_original.mp4](V1_vs_original.mp4) | 124 / 5.167 | Original H3-World vs V1 | A / D |
| [V2_A_vs_original.mp4](V2_A_vs_original.mp4) | 124 / 5.167 | Original H3-World vs V2 | action A |
| [V2_D_vs_original.mp4](V2_D_vs_original.mp4) | 124 / 5.167 | Original H3-World vs V2 | action D |
| [V2_vs_original.mp4](V2_vs_original.mp4) | 124 / 5.167 | Original H3-World vs V2 | A / D |
| [V2_A_vs_V1.mp4](V2_A_vs_V1.mp4) | 124 / 5.167 | V1 vs V2 | action A |
| [V2_D_vs_V1.mp4](V2_D_vs_V1.mp4) | 124 / 5.167 | V1 vs V2 | action D |
| [V2_vs_V1.mp4](V2_vs_V1.mp4) | 124 / 5.167 | V1 vs V2 | A / D |
| [V3_A_vs_original.mp4](V3_A_vs_original.mp4) | 56 / 2.333 | Original H3-World vs V3 | action A |
| [V3_D_vs_original.mp4](V3_D_vs_original.mp4) | 56 / 2.333 | Original H3-World vs V3 | action D |
| [V3_vs_original.mp4](V3_vs_original.mp4) | 56 / 2.333 | Original H3-World vs V3 | A / D |
| [V3_A_vs_V2.mp4](V3_A_vs_V2.mp4) | 56 / 2.333 | V2 vs V3 | action A |
| [V3_D_vs_V2.mp4](V3_D_vs_V2.mp4) | 56 / 2.333 | V2 vs V3 | action D |
| [V3_vs_V2.mp4](V3_vs_V2.mp4) | 56 / 2.333 | V2 vs V3 | A / D |
| [V3_four_paths_56.mp4](V3_four_paths_56.mp4) | 56 / 2.333 | V3 C12->5: A->A / A->D / D->A / D->D |
| [V1_vs_V0.mp4](V1_vs_V0.mp4) | 124 / 5.167 | V0→V1；与Original对照字节相同 |

V3主对照只显示Original/V2的RGB[0,56)；两者完整124f源片在各版本videos保留。V3四路径在RGB39高亮chunk边界。A→D/D→A没有匹配Original/V2视频，标为MISSING_MATCHED_VIDEO，不做假配对。

[对比公平性](../COMPARISON_PROTOCOL.md) · [版本变化与未解决问题](../roadmap.md) · [返回汇报](../README.md)
