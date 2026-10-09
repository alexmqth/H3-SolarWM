# H3-World × SolarWM：一分钟汇报导航

**目标：** 将SolarWM的因果分块、KV cache与少步方法迁入H3-World，同时保留action control与画面连续性，最终改善长视频效率。

**当前结论：** EXP-001 / v3已验收：单停车场、seed13，持续A/D六窗口124帧（5.17秒）具备可辨响应与基本人物/场景结构。保留AA RGB72→73姿态跳变、动作节奏不均和局部细节软化；DD后段靠近画面下边缘。AD/DA仅到73帧，DA第三块flow轻微反号；切换和泛化未通过。无persistent KV、无公平加速结论，V3仍计划中。 V2a仍是并列的RGB视觉适配路线，不能把两支结果相加成V3。

**先看最新交付：** [Original vs V2b持续A/D完整124帧总览](V2b_same_sigma_local_bidir/videos/Original_vs_V2b_AD_overview_248.mp4)。

**建议播放顺序：**

1. [V0 vs V1：直接因果化的代价](00_comparison_gallery/V1_vs_original.mp4)。
2. [V1 vs V2a：RGB联合视觉修复](00_comparison_gallery/V2a_vs_V1.mp4)。
3. [V1 vs V2b：恢复原生条件与局部联合重算](00_comparison_gallery/V2b_vs_V1.mp4)。
4. [V2a vs V2b：并列路线的能力取舍](00_comparison_gallery/V2a_vs_V2b.mp4)，采样、history和KV不同，非单变量消融。
5. [V2b四路径56f](00_comparison_gallery/V2b_four_paths_56.mp4)，RGB39高亮第二块；[V2a完整20秒失败片](V2a_rgb_anchor/videos/V2a_long20s_failure.mp4)保留后段。

| 版本 | Action fidelity | Visual stability | 已验证范围 | Persistent KV | 采样 | 新增训练 | 当前结论 |
|---|---|---|---|---|---|---|---|
| V0 Original | A/D方向正控 | 124f基本完整 | 124f及已有长片参考 | 否 | 30整段；另存50步 | 无，released LoRA | Reference，非GT |
| V1 Native causal | 显著下降 | 本代表停滞、过亮/背景退化；其他早期协议有重影 | 124f工程rollout | 是，CPU raw video KV | 8/chunk | 本代表无 | 工程可行，联合质量失败 |
| V2a RGB-Anchor | A/D方向失败 | **124f人物结构相对稳定**；20s失败 | 124f视觉证据；243/481f负结果 | 是，clean commit的历史hidden KV | 8/chunk | visual adapter +已训action residual | Longer-horizon Visual Stability Demonstrated（仅124f scope） |
| V2b Same-σ local bidir | 持续A/D在124f有可辨响应；切换仍有限制 | 124f人物/场景基本可用；边界与节奏有缺陷 | 单停车场、seed13、六块124f | **否**，每步重算全部可见历史 | 30/chunk | **无**，Original + released LoRA | Sustained A/D feasibility accepted；非V3 |
| V3 Efficient causal | 目标：保留 | 目标：保留并扩大范围 | **尚未实现** | 目标：strict causal + KV | 先可信30步，后AnyFlow | 待定 | Planned Unification，无模型/视频 |

## 按需展开

- [V0](V0_original/README.md) / [V1](V1_native_causal/README.md) / [V2a](V2a_rgb_anchor/README.md) / [V2b](V2b_same_sigma_local_bidir/README.md)：统一十项说明、代码快照、真实原片。
- [比较/参考画廊](00_comparison_gallery/README.md) / [本地浏览器演示页](index.html)。
- [并列路线图与维度对照](roadmap.md) / [研究分支](01_research_branches_summary/README.md) / [5分钟讲稿](TALK_5MIN.md)。
- [未来V3与验收](02_next_steps/README.md) / [历史指标](METRICS.md) / [公平性与缺失](COMPARISON_PROTOCOL.md)。

EXP-001 / v3已验收：单停车场、seed13，持续A/D六窗口124帧（5.17秒）具备可辨响应与基本人物/场景结构。保留AA RGB72→73姿态跳变、动作节奏不均和局部细节软化；DD后段靠近画面下边缘。AD/DA仅到73帧，DA第三块flow轻微反号；切换和泛化未通过。无persistent KV、无公平加速结论，V3仍计划中。

本目录可独立复制汇报：MP4均为真实文件，核心code供阅读，不含33B权重。当前标签与文件名均使用V2a/V2b；旧V2/V3编号片保留在完整仓库archive内。最新EXP-001包含受控模型推理、0训练；原56帧素材完整保留。
