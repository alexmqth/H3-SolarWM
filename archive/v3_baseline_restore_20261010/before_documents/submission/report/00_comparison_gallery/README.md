# 最新：V2c可行性版本

[Original vs V2c AA124](../v2/v2c_strict_causal_kv/videos/Original_vs_V2c_AA_124.mp4) · [V2b vs V2c AA124](../v2/v2c_strict_causal_kv/videos/V2b_vs_V2c_AA_124.mp4) · [V2c AD124原片](../v2/v2c_strict_causal_kv/videos/V2c_AD_124.mp4)。保留AA瞬态明显人体形变与质量限制，详情见[V2c](../v2/v2c_strict_causal_kv/README.md)。

---

# 当前比较画廊：V2a / V2b并列路线

这里是当前正式标签的25条H.264/24fps视频。V2a与V2b各自对照V0、V1，再进行两支能力比较；它们不是先后升级或checkpoint继承。

**最新124帧交付：** [持续A](Original_vs_V2b_A_124.mp4) / [持续D](Original_vs_V2b_D_124.mp4) / [A/D顺序总览248帧](Original_vs_V2b_AD_overview_248.mp4)。EXP-001 / v3已验收：单停车场、seed13，持续A/D六窗口124帧（5.17秒）具备可辨响应与基本人物/场景结构。保留AA RGB72→73姿态跳变、动作节奏不均和局部细节软化；DD后段靠近画面下边缘。AD/DA仅到73帧，DA第三块flow轻微反号；切换和泛化未通过。无persistent KV、无公平加速结论，V2c已由EXP-002/003完成124帧有限可行性验收。

**先看两支取舍：** [V2a vs V2b](V2a_vs_V2b.mp4)。左V2a：8步、clean commit/CPU KV、RGB+visual adapter；右V2b：30步、Same-σ局部双向重算、Single I0、无新增训练/无persistent hidden KV。仅比较共同前56帧，非单变量消融。

| 视频 | 帧数 / 秒 | 比较范围 |
|---|---:|---|
| [V0_AD_reference.mp4](V0_AD_reference.mp4) | 124 / 5.167 | V0 action reference: A vs D |
| [V0_WSAD_reference.mp4](V0_WSAD_reference.mp4) | 124 / 5.167 | V0 Original H3-World: W / S / A / D |
| [V1_A_vs_original.mp4](V1_A_vs_original.mp4) | 124 / 5.167 | Original H3-World vs V1 | action A |
| [V1_D_vs_original.mp4](V1_D_vs_original.mp4) | 124 / 5.167 | Original H3-World vs V1 | action D |
| [V1_vs_original.mp4](V1_vs_original.mp4) | 124 / 5.167 | Original H3-World vs V1 | A / D |
| [V1_vs_V0.mp4](V1_vs_V0.mp4) | 124 / 5.167 | V0 vs V1，Original对照的同义命名副本 |
| [V2a_A_vs_original.mp4](V2a_A_vs_original.mp4) | 124 / 5.167 | V0 vs V2a | action A |
| [V2a_D_vs_original.mp4](V2a_D_vs_original.mp4) | 124 / 5.167 | V0 vs V2a | action D |
| [V2a_vs_original.mp4](V2a_vs_original.mp4) | 124 / 5.167 | V0 vs V2a | A / D |
| [V2a_A_vs_V1.mp4](V2a_A_vs_V1.mp4) | 124 / 5.167 | V1 vs V2a | action A |
| [V2a_D_vs_V1.mp4](V2a_D_vs_V1.mp4) | 124 / 5.167 | V1 vs V2a | action D |
| [V2a_vs_V1.mp4](V2a_vs_V1.mp4) | 124 / 5.167 | V1 vs V2a | A / D |
| [V2b_A_vs_original.mp4](V2b_A_vs_original.mp4) | 56 / 2.333 | V0 vs V2b | action A |
| [V2b_D_vs_original.mp4](V2b_D_vs_original.mp4) | 56 / 2.333 | V0 vs V2b | action D |
| [V2b_vs_original.mp4](V2b_vs_original.mp4) | 56 / 2.333 | V0 vs V2b | A / D |
| [V2b_A_vs_V1.mp4](V2b_A_vs_V1.mp4) | 56 / 2.333 | V1 vs V2b | action A |
| [V2b_D_vs_V1.mp4](V2b_D_vs_V1.mp4) | 56 / 2.333 | V1 vs V2b | action D |
| [V2b_vs_V1.mp4](V2b_vs_V1.mp4) | 56 / 2.333 | V1 vs V2b | A / D |
| [V2a_vs_V2b_A.mp4](V2a_vs_V2b_A.mp4) | 56 / 2.333 | V2a vs V2b | PARALLEL research approaches | action A |
| [V2a_vs_V2b_D.mp4](V2a_vs_V2b_D.mp4) | 56 / 2.333 | V2a vs V2b | PARALLEL research approaches | action D |
| [V2a_vs_V2b.mp4](V2a_vs_V2b.mp4) | 56 / 2.333 | V2a vs V2b | PARALLEL research approaches | A / D |
| [V2b_four_paths_56.mp4](V2b_four_paths_56.mp4) | 56 / 2.333 | V2b local C12->5: A->A / A->D / D->A / D->D |

V2b四路径在RGB39高亮第二块。Original/V1/V2a是124f源片裁前56f；完整原片仍在各版videos中。没有匹配RGB39切换的Original/V2a A→D或D→A视频，四路径是V2b内部证据，未伪造配对。

V2a仅有124f视觉相对稳定证据，20秒失败；V2b新增持续A/D124f可行性证据，仍不能声称跨场景或10/20秒长期稳定。旧编号V2/V2c片已移出report归档，本目录不混用旧标签。

[并列路线与结论](../roadmap.md) · [公平性](../COMPARISON_PROTOCOL.md) · [首页](../README.md)
